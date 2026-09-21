"""Study F: FastHenry as an INDEPENDENT magnetoquasistatic referee for the FDFD
spiral extraction -- and the re-judgement of the 2.5-4.6 % residual that the
level-invariant ladder (study P, ``invariant_ladder.py``) could not explain.

ANSWER, up front: the residual is the GREENHOUSE REFEREE'S error, not the
FDFD's. FastHenry meshes the SAME conductors as a 3-D PEEC volume and SOLVES
for the current distribution instead of assuming it uniform. Its continuum
limits are

    L(FastHenry, the FIXTURE the FDFD measures)      = 319.913 +- 0.151 pH
    L(FastHenry, between the planes, physical bridge) = 319.182 +- 0.157 pH
    L(FastHenry, the REFEREE's quantity)             = 318.743 +- 0.163 pH
    L(Greenhouse referee)                            = 333.055 pH
    FDFD Richardson range = [317.874, 324.806] pH   (CONTAINS all three)

so the referee is +4.10 % / +4.34 % / +4.49 % above them, and on ITS OWN
quantity -- the strip minus a bridge with no vertical member and no ground
contact, which is what its 333.055 pH is built from -- its error is -14.313 pH
= -4.297 %. Of the FDFD's -2.48 % to -4.56 % gap against that referee, -3.95 %
is the referee being high (like quantity against like: the FastHenry fixture
limit against the referee) and the rest is the spread of the FDFD's own
extrapolation: the FDFD range BRACKETS this referee, its two ends -0.64 %
below and +1.53 % above the FastHenry fixture value (equivalently the FastHenry
limit sits +0.65 % / -1.50 % from the two ends, which is how the gate reads
it), INSIDE 3 % and INSIDE 5 %. The referee's -14.313 pH decomposes (measured,
each term a FastHenry limit minus the Greenhouse value of the SAME conductor,
and the pieces close to 0.0001 pH): -7.85 pH the eight right-angle corners in
situ, -6.33 pH the via and the underpass reversal, -0.12 pH the referee's
post-less bridge. Two further terms are fixture physics rather than referee
error and are reported apart from it: the short standard's post REACHING the
ground plane (-0.438 pH on the between-planes quantity) and the lead columns
(+0.730 pH, not part of the referee's quantity at all).

WHY FASTHENRY IS AN INDEPENDENT REFEREE. It is the MIT PEEC/multipole
inductance extractor (3.0wr branch, pinned and built by
``validation/referees/fasthenry/build_fasthenry.sh``; commit, licence, the one
printf patch and the binary's own SHA-256 are recorded there and in this
study's JSON under ``fasthenry`` and ``cost``). It shares the Hoer-Love
partial-inductance FORMULA with the Greenhouse referee -- and nothing else: it
is 1990s C, it discretises the metal into filaments, and it SOLVES a
mesh-analysis system for the current distribution (R + j omega L), which is
exactly the assumption the Greenhouse sum makes by hand. It is not the FDFD
either: an integral-equation code in open space, no grid outside the metal, no
walls, no PML, no dielectrics, no capacitance. The agreement below is therefore
between three genuinely different methods, and the disagreement is
attributable. Every cached run is keyed on the SHA-256 of the binary as well as
the deck, so no number here (and no "live" gate in the test file) can be
replayed from a cache written by another build.

THE GROUND: THE IMAGE CONSTRUCTION (exact for an infinite PEC plane). Every
conductor is mirrored in z = 0 and driven as a SECOND port with -I; the
mirror-antisymmetric solution has E_t = 0 on the plane, so its restriction to
z > 0 solves the half-space problem exactly (conductors that TOUCH the ground
-- the lead columns, the short standard's grounded post -- are included: the
real and image halves simply JOIN there, one shared node per cell face at
z = 0 with the real cell and its mirror on it, which is the monopole-over-
ground equivalence). L = Im(Z11 - Z12)/omega for an open path; for the whole
fixture, whose real and image halves close one loop, L is HALF the loop's (the
two halves store equal energy). FastHenry's own ground-plane element is NOT
used. Validated on one bar 26 um over the plane against the Greenhouse image
sum: 4.9e-8 relative (F1).

THE TWO CONDUCTOR MODELS, and why both are needed
-------------------------------------------------
* SEGMENT model -- what a designer writes by hand: one FastHenry segment per
  centreline piece with ``nwinc x nhinc`` filaments in it, segments meeting at
  the centreline corner POINT. At low frequency its filaments carry equal
  current densities, so it IS the Greenhouse uniform-current bar model. It
  reproduces the Greenhouse sum of the same bars to 1.0e-9 (345.5475 pH,
  strip; F2) -- which is this study's PLUMBING gate: geometry, ports, image
  construction and de-embedding all agree with the referee bar for bar.
* MESH model -- a 3-D PEEC discretisation of the metal VOLUME: a rectilinear
  grid through the conductor, a node at every cell centre and at every internal
  face, and one segment from each cell centre to each face with exactly that
  face's cross-section. For a uniform flow the segments tile the metal exactly
  once, so a straight run costs NOTHING (splitting a uniform bar changes no
  partial inductance: the 200 um bar is the closed form to 5.7e-7 at every
  level, F5); where the current TURNS -- corners, the via, the transition out
  of a lead column, the grounded post -- the cells let it redistribute. This is
  the model that answers F3/F4/F5.

THE FIXTURE, exactly the one the FDFD solves (gate F0, from
``invariant_ladder.json``'s own read-back of its BUILT cell masks): the 2-turn
spiral (r_out 52 um, W = S = 10 um, M2 2 um thick, ``rect_spiral`` centreline
taken from ``rfx.fdfd.gds`` and checked against ``spiral_greenhouse``'s
independent copy to 1.7e-21 m), the via, the M1 underpass 4 um below the M2
centre line, the two lead columns, the 10 um lumped port gaps, the short
standard's M2 arm + grounded post + M1 arm, and the PEC ground 26 um below the
M2 centre. Metal volumes 15840 / 2600 / 6100 um^3 (DUT / open / short) and
every object box reproduce the FDFD's to 2.2e-16 and 0 m.

THREE DE-EMBEDDED QUANTITIES are reported, because three different things can
be meant by "the de-embedded inductance" and only one of them is the
referee's. The two steps between them are MEASURED here (F3):
* ``planes_referee``  = L(strip) - L(bridge with the post truncated to the
  metal stack), between the two reference planes the open/short de-embedding
  leaves (y = -57 um on the strip, x = 47 / 7 um on the bridge -- the
  lead-column footprint centres). The Greenhouse bridge is two HORIZONTAL bars
  with no vertical member and no ground contact, so this is the referee's
  quantity conductor for conductor and F3 compares it with 333.055 pH.
* ``planes_physical`` = the same with the fixture's PHYSICAL bridge, whose post
  reaches z = 0 and joins its own image there. That ground contact is a second
  real-to-image path, not a modelling approximation, and it is worth
  -0.438 pH (-0.137 %) on this quantity (the bridge itself goes 16.439 ->
  16.001 pH), so it is reported as its own term instead of being charged to the
  referee.
* ``fixture``         = L(DUT) - L(SHORT) of the WHOLE fixture driven
  differentially, port gaps and lead columns included. This is what the FDFD's
  open/short de-embedding measures, so F4 compares this one; it is
  +0.730 pH (+0.229 %) above ``planes_physical``, which is the transition out
  of the lead columns. The DUT is a two-terminal element in magnetoquasistatics
  (no displacement current: whatever enters the outer port leaves the inner
  one), so its differential mode is realised with ONE port and the inner
  crossing shorted; the SHORT standard's grounded post is a second real-to-
  image path, so it gets two ports and the honest Z11 - Z12 - Z21 + Z22.

RESULTS (every number is in ``fasthenry_referee.json``)
--------------------------------------------------------
F0  PASSES. Structural, above.
F1  PASSES. Closed forms, worst 3.7e-6 against a 1e-3 gate, after the filament
    ladder (nwinc x nhinc = 1x1, 2x2, 4x4, 8x4): Hoer-Love partial SELF of a
    100 x 10 x 2 um bar 1.5e-13 at one filament and 3.0e-9 at 8x4, a 10 um cell
    5.3e-9, the 4 um via bar 2.4e-15; MUTUALS of two 100 um bars, coplanar one
    width apart 3.3e-9 (30.268 pH), stacked at the underpass offset 1.4e-11
    (53.777 pH), collinear with a 50 um gap 4.5e-9 (7.273 pH), ten widths apart
    3.7e-6 (9.355 pH -- the one number where FastHenry's far-field filament
    approximation shows); the IMAGE construction 4.9e-8.
F2  PASSES. Filament convergence, two knobs, at the low-frequency limit and at
    the FDFD protocol's 100 MHz / sigma = 2e6 S/m alike, and the extrapolation
    checked against a level the ladder does not use.
    * SEGMENT model: 345.5475 pH at every (nwinc, nhinc, subdivision along the
      length) in the ladder -- spread 5.4e-7 at 1 kHz and 5.2e-7 at 100 MHz,
      and the two frequencies agree to 2.2e-16. The uniform-current limit is
      exact in this model, so a flat ladder is the expected result and its
      value is the gate on the plumbing (1.0e-9 from the Greenhouse sum of the
      same bars plus the via bar and its ground image).
    * MESH model, cells across the strip width k = 1, 2, 3, 4, 6, 8:
      L(strip) = 344.1661 / 338.3283 / 336.8831 / 336.2892 / 335.8004 /
      335.6031 pH, monotone and contracting, observed order 1.601, LIMIT
      335.180 +- 0.169 pH. The +- is half the SPREAD of the p = 1 / p = 2 /
      observed-order estimates: a disagreement between estimators, NOT a bound
      on the discretisation error.
    * THE EXTRAPOLATION ITSELF (``confirmation``): one level FINER than the
      ladder, k = 10, on the strip (335.5019 pH, 34768 segments, 410 s) and on
      both bridges, then re-extrapolated from the three finest points:
      335.225 +- 0.128 pH for the strip (observed order 2.324) against the
      ladder's 335.180 +- 0.169, i.e. 0.028 pH apart against a 0.315 pH
      tolerance; 318.784 +- 0.123 against 318.743 +- 0.163 for
      ``planes_referee`` and 319.221 +- 0.119 against 319.182 +- 0.157 for
      ``planes_physical``. The quoted spreads were honest, and they are
      conservative: every re-extrapolation lands inside a fifth of one.
    * At 100 MHz the mesh ladder differs by -3.5e-7 relative at its finest
      common level, and the frequency ladder at k = 3 is flat from 1 kHz to
      100 MHz to 3.1e-7 (336.8831 pH at every decade): the uniform-current
      PLATEAU the FDFD protocol assumes is real, and FastHenry has no
      capacitance, so none of the FDFD's RC contamination is in these numbers.
    * Discretisation sensitivities at k = 3 (relative to 336.8831 pH): the cell
      size along a run halved -6.7e-5, doubled +1.5e-4; two cells across the
      2 um metal slab -6.7e-5; 2x2 filaments inside every mesh segment -4.3e-7.
      All far below the 4.3 % the answer is about.
F3  PASSES (the difference is RESOLVED: 80 x its own spread).
    ``planes_referee``  = 327.4358 / 321.7598 / 320.3763 / 319.8083 /
    319.3396 / 319.1496 pH at k = 1..8, observed order 1.591, LIMIT
    318.743 +- 0.163 pH, i.e. -14.313 pH = -4.297 % against the Greenhouse
    referee's 333.055 pH.
    ``planes_physical`` = 327.4358 / 322.0800 / 320.7568 / 320.2110 /
    319.7592 / 319.5754 pH, order 1.581, LIMIT 319.182 +- 0.157 pH = -4.166 %.
    ``fixture``         = 327.7412 / 322.6753 / 321.4200 / 320.9002 /
    320.4681 / 320.2915 pH, order 1.568, LIMIT 319.913 +- 0.151 pH = -3.946 %.
    The two steps: the post's ground contact -0.438 pH (-0.137 %) and the
    lead-column transition +0.730 pH (+0.229 %), the latter a term the referee
    cancels by a 90-degree rotation symmetry and the metal does not. (At the
    COARSEST level the two bridges coincide to 1e-14: with one cell across the
    post the two arms are a balanced bridge and the single branch through z = 0
    carries no current. The branch opens as soon as the current can
    redistribute laterally, and the difference then converges monotonically,
    0.320 / 0.380 / 0.403 / 0.420 / 0.426 pH at k = 2, 3, 4, 6, 8.)
F4  PASSES, and the V1 verdict is REVERSED. Against the FDFD's Richardson range
    [317.874, 324.806] pH the FastHenry fixture limit 319.913 pH sits INSIDE,
    at +0.65 % / -1.50 % from the two ends. The only model differences between
    that quantity and the FDFD's ``L_dut`` are the two study P measured and did
    not apply -- the 10 W walls +0.186 % and the RC contamination of
    Im Z / omega +0.095 % -- and with them the range becomes [318.766,
    325.718] pH and the distances +0.36 % / -1.78 %. Study P's THIRD
    correction, the post short (-0.33 to -0.36 %, from its straight-bar thru
    probe), is read and NOT applied: it converts ``L_dut`` into a third
    quantity, while this study measures the two steps from ``fixture`` to the
    referee's quantity itself (above), and using both would double-count. Each
    FastHenry quantity is compared with the FDFD range carried to it by those
    measured steps ([318.039, 324.975] pH for ``planes_physical``, [317.603,
    324.529] pH for ``planes_referee``; the distances are then +0.36 % /
    -1.78 % for all three, since the conversion is one factor). Both gates
    (3 % and 5 %) pass.
    WHAT IS EXTRAPOLATED AND WHAT IS MEASURED (``extrapolation_honesty``).
    "The FDFD range contains the FastHenry limit" compares two extrapolations,
    so the FDFD side is also reported raw: its finest MEASURED level (m = 4,
    312.675 pH) is still -2.267 % from the FastHenry fixture limit and its
    ladder order is still rising (1.447 -> 1.627), while its three individual
    limit estimates are +1.525 % (p = 1), -0.642 % (p = 2) and -0.149 %
    (observed order, 319.451 pH) from it. The observed-order estimate -- the
    one whose exponent that study measured rather than assumed -- agrees with
    this referee to 0.15 %, which is far better than the +-1 % width of the
    range implies; the honest statement is that the FDFD's extrapolated limit
    is right to about 1.8 % on the pessimistic member of its own spread and to
    0.15 % on the member its data support, while its finest measured level is
    still 2.3 % low.
F5  PASSES. D2c's fixtures (W = 10 um, t = 2 um, l1 = l2 = 100 um, ground
    26 um below the M2 centre), between the same reference planes:
    * the straight 200 um bar is 109.6093 pH at every level -- the Hoer-Love
      image value to 5.7e-7 (uniform current, so the mesh model is exact);
    * the L-bend is 101.8254 / 101.2873 / 101.1078 / 101.0686 / 101.0540 /
      101.0430 / 101.0391 / 101.0362 pH at k = 1, 2, 4, 6, 8, 12, 16, 24,
      LIMIT 101.032 +- 0.002 pH against the Greenhouse 101.8310 pH: ONE
      right-angle corner costs -0.799 pH more than the area-exact
      uniform-current bars say;
    * the corner EXCESS L(bend) - L(bar) is -7.7839 / -8.3220 / -8.5014 /
      -8.5407 / -8.5554 / -8.5663 / -8.5703 / -8.5731 pH, LIMIT
      -8.577 +- 0.002 pH (observed order 1.948 -- this ladder converges at
      first order, which is why it is carried to k = 24) against the referee's
      -7.778 pH: same sign, same mechanism (perpendicular halves do not
      couple, collinear ones do), magnitude 1.103 x the referee's.
    So FastHenry does reproduce the corner excess -- and it re-judges D2c's K3
    too: that study read its FDFD excess (-7.261 pH at one cell across W,
    -7.653 pH at two, wall-corrected) as approaching the referee's -7.78 pH to
    1.7 %, but the true value is -8.58 pH, so the FDFD at two cells was 11 %
    SHORT of it and still converging, not converged.

WHERE THE REFEREE'S -14.313 pH COMES FROM (``attribution``; an identity, not a
fit, on the referee's own quantity). L(strip) - L(post-less bridge) splits into
the strip's error, -14.191 pH (-4.06 %: 335.198 pH against 349.371 pH), minus
the bridge's, +0.123 pH (+0.75 %: 16.439 pH against 16.316 pH -- the referee's
bridge is LOW, because the referee has no vertical member where the metal has
one, exactly as it has no via); the two close on the directly measured total to
0.0001 pH. The strip's error splits again, on its own ladder:
* CORNERS IN SITU, the M2 spiral alone between the outer plane and the inner
  end of the strip (no via, no underpass): 335.848 +- 0.153 pH against
  343.710 pH = -7.862 pH. The isolated-corner cross-check gives
  8 x (-0.799) = -6.390 +- 0.014 pH -- 19 % smaller, because an isolated corner
  has no neighbouring turns for its redistributed current to couple to.
* VIA AND UNDERPASS, the rest: -6.329 pH. The isolated inner-end probe (M2
  extension + via + underpass, 13.242 +- 0.017 pH against the referee's two
  antiparallel bars with no via, 14.913 pH) gives -1.672 pH of that on its own,
  and its flat control (the M2 extension alone) reproduces the closed form to
  7.5e-9 -- so the -1.672 pH is the via and the reversal, and the remaining
  -4.7 pH is how that changed current couples to the rest of the spiral.
Seen the other way: of the strip's -14.191 pH, -3.824 pH is the referee's
CORNER CONVENTION (its area-exact tiling against FastHenry's centreline
segments, both uniform-current) and -10.350 pH (-3.00 %) is the uniform-current
ASSUMPTION itself -- the mesh limit minus the segment model on the same
conductor.

WHAT THIS DOES NOT CLAIM. FastHenry is magnetoquasistatic: no capacitance, no
substrate loss, no radiation, and its metal here carries the FDFD protocol's
sigma = 2e6 S/m with the skin depth (35.6 um) well above every dimension, so
"L" means the same DC-internal-inductance quantity in all three tools. The
walls, the lid and the dielectrics of the FDFD fixture have no counterpart
here; study P's own measurements of what they are worth (+0.186 % and
+0.095 %) are applied to ITS range, not to these numbers. Every limit here is
a Richardson extrapolation of a six-level ladder (eight for the bend) whose
quoted +- is the spread of the p = 1 / p = 2 / observed-order estimates and not
a confidence interval or an error bound -- which is what ``confirmation``
measures rather than asserts (F2).

COST (this machine, one core; ``cost`` in the JSON records every deck, its tag
and its wall clock): the dense mesh solve (``-s ludecomp -m direct -a off``: no
multipole approximation, no automatic filament refinement) costs 128.8 s at
22288 segments for the strip at k = 8, 321.8 s at 31440 for the whole DUT and
409.8 s at 34768 for the strip at the confirmation level k = 10; the whole
study is 96 decks and 1632.7 s of FastHenry, cached under
``$RFX_REFEREE_CACHE`` (default ``~/.cache/rfx-referees``) by the SHA-1 of the
deck, the solver options and the SHA-256 of the binary, so a re-run with this
build costs nothing, a re-run with any other build re-measures everything, and
an interrupted study loses nothing (every block, and every fixture's ladder, is
written to the JSON as soon as it is measured).

Run::

    sh validation/referees/fasthenry/build_fasthenry.sh      # once
    .venv/bin/python validation/fdfd/fasthenry_referee.py [--ks 1 2 3 4 6 8]
        [--confirm-k 10] [--from-json] [--fresh] [--no-figure]

writes ``fasthenry_referee.json`` and ``fasthenry_referee.png`` next to this
file. ``--from-json`` re-derives the gates from the JSON and runs nothing.

Figure: (a) the three de-embedded ladders against h/W with the Greenhouse line,
the FDFD Richardson band and the off-ladder k = 10 points (hollow); (b) the F1
closed-form residuals against the 0.1 % gate; (c) the corner excess ladder
against the referee's; (d) the uniform-current plateau from 1 kHz to 100 MHz;
(e) the decomposition of the referee's error, with the two terms that are not
its error in grey; (f) cost against segment count.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import pathlib
import shutil
import subprocess
import sys
import time
from typing import Any, Callable, Iterable, Sequence

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
HERE = pathlib.Path(__file__).resolve().parent
REFEREE_PATH = REPO / "validation" / "crossval" / "comparators" / "spiral_greenhouse.py"
JSON_PATH = HERE / "fasthenry_referee.json"
PNG_PATH = HERE / "fasthenry_referee.png"
INVARIANT_LADDER_PATH = HERE / "invariant_ladder.json"
BUILD_SCRIPT = REPO / "validation" / "referees" / "fasthenry" / "build_fasthenry.sh"

MU0 = 4e-7 * math.pi

# ---- the fixture, copied from validation/fdfd/invariant_ladder.py ----------
FREQ = 1e8                 # the FDFD protocol frequency
SIGMA = 2e6                # S/m, the FDFD protocol's volumetric metal
FREQ_LOW = 1e3             # the uniform-current (DC) limit, see the f ladder
N_TURNS = 2
R_OUT, SPACING, WIDTH = 52e-6, 10e-6, 10e-6
LEAD = 1.5 * WIDTH
T_SI, T_OX_LOW, T_M1, T_VIA, T_M2 = 20e-6, 1e-6, 2e-6, 2e-6, 2e-6
Z_M1 = T_SI + T_OX_LOW                       # 21 um, M1 bottom
Z_VIA = Z_M1 + T_M1                          # 23 um
Z_M2 = Z_VIA + T_VIA                         # 25 um, M2 bottom
Z_M2_TOP = Z_M2 + T_M2                       # 27 um
GROUND_H = Z_M2 + 0.5 * T_M2                 # 26 um, ground -> M2 centre line
DZ_UNDER = 0.5 * T_M1 + T_VIA + 0.5 * T_M2   # 4 um, M2 centre -> M1 centre
PORT_GAP = 10e-6                             # column bottom (lumped port gap 0 .. 10 um)
SHORT_GAP = WIDTH                            # physical isolating gap column -> post
REF_SHIFT = 0.5 * WIDTH                      # reference plane at the column footprint CENTRE

# corner study D2c (validation/fdfd/corner_convergence.py), same stack
BEND_L1 = BEND_L2 = 100e-6
BAR_LENGTH = BEND_L1 + BEND_L2

# the numbers this study is judged against (quoted, re-derived where possible)
GREENHOUSE_DEEMBEDDED = 3.3305510896906523e-10     # invariant_ladder.json referee.total
FDFD_LIMIT_RANGE = (3.178738461268806e-10, 3.2480620346382214e-10)   # gates.P2.range

FH_OPTS = ("-s", "ludecomp", "-m", "direct", "-a", "off")


# ----------------------------------------------------------------------------
# 0. the Greenhouse referee and the FastHenry binary

def load_referee() -> Any:
    spec = importlib.util.spec_from_file_location("spiral_greenhouse_for_f", REFEREE_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def fasthenry_binary() -> pathlib.Path | None:
    """The built referee binary: ``$RFX_FASTHENRY``, the build script's cache
    (``$RFX_REFEREE_CACHE`` or ``~/.cache/rfx-referees``), or ``PATH``."""
    env = os.environ.get("RFX_FASTHENRY")
    if env:
        p = pathlib.Path(env)
        return p if p.is_file() and os.access(p, os.X_OK) else None
    cache = pathlib.Path(os.environ.get("RFX_REFEREE_CACHE",
                                       pathlib.Path.home() / ".cache" / "rfx-referees"))
    p = cache / "fasthenry" / "bin" / "fasthenry"
    if p.is_file() and os.access(p, os.X_OK):
        return p
    w = shutil.which("fasthenry")
    return pathlib.Path(w) if w else None


def manifest() -> dict[str, Any]:
    cache = pathlib.Path(os.environ.get("RFX_REFEREE_CACHE",
                                       pathlib.Path.home() / ".cache" / "rfx-referees"))
    p = cache / "fasthenry" / "manifest.json"
    if not p.is_file():
        return {}
    try:
        man = json.loads(p.read_text())
    except json.JSONDecodeError:
        return {}
    d = man.get("patch_diff_file")
    if d and pathlib.Path(d).is_file():
        man["patch_diff"] = pathlib.Path(d).read_text()
    return man


# ----------------------------------------------------------------------------
# 1. geometry: axis-aligned boxes, in metres, in the FDFD's own coordinates
#    (ground plane at z = 0, M2 slab 25 .. 27 um, the spiral of
#    rfx.fdfd.gds.rect_spiral with lead = 1.5 W)

Box = tuple[float, float, float, float, float, float]   # x0, x1, y0, y1, z0, z1


def spiral_centreline() -> np.ndarray:
    """Centreline of ``rfx.fdfd.gds.rect_spiral(N_TURNS, R_OUT, WIDTH, SPACING,
    LEAD)`` -- taken from ``rfx`` (the geometry MUST be the FDFD's) and gated
    against the independent copy in ``spiral_greenhouse`` by :func:`geometry_gate`."""
    from rfx.fdfd import gds
    sp = gds.rect_spiral(N_TURNS, R_OUT, WIDTH, SPACING, LEAD)
    return np.asarray(sp.centreline, dtype=np.float64)


def strip_rectangles(pts: np.ndarray, width: float = WIDTH) -> list[tuple[float, float, float, float]]:
    """The area-exact tiling of the drawn strip polygon: one rectangle per
    centreline segment, each translated by ``+width/2`` along its own direction
    except the start of the first and the end of the last (the convention of
    ``spiral_convergence.area_exact_chain``).  Consecutive rectangles then touch
    without overlapping and their union IS the polygon -- which is all this
    function is used for: the rectangles are a description of the metal region,
    not a current model (the mesh below tiles that region with cells)."""
    hw = 0.5 * width
    out = []
    n = len(pts) - 1
    for i in range(n):
        a = np.asarray(pts[i], float)
        b = np.asarray(pts[i + 1], float)
        d = b - a
        ell = float(np.hypot(*d))
        u = d / ell
        sa = a if i == 0 else a + hw * u
        sb = b if i == n - 1 else b + hw * u
        lo = np.minimum(sa, sb) - hw * np.abs(np.array([u[1], u[0]]))
        hi = np.maximum(sa, sb) + hw * np.abs(np.array([u[1], u[0]]))
        out.append((float(lo[0]), float(hi[0]), float(lo[1]), float(hi[1])))
    return out


def spiral_boxes() -> list[Box]:
    """The M2 strip as boxes (the area-exact rectangles x the M2 slab)."""
    return [(r[0], r[1], r[2], r[3], Z_M2, Z_M2_TOP) for r in strip_rectangles(spiral_centreline())]


def inner_x() -> tuple[float, float]:
    a0 = R_OUT - 0.5 * WIDTH
    a_in = a0 - (WIDTH + SPACING) * N_TURNS
    return a_in - 0.5 * WIDTH, a_in + 0.5 * WIDTH      # 2 .. 12 um


def outer_x() -> tuple[float, float]:
    a0 = R_OUT - 0.5 * WIDTH
    return a0 - 0.5 * WIDTH, a0 + 0.5 * WIDTH          # 42 .. 52 um


def port_line_y() -> tuple[float, float]:
    """The lead-column footprint rows: ``y in [-a0 - lead, -a0 - lead + W]``."""
    a0 = R_OUT - 0.5 * WIDTH
    y0 = -a0 - LEAD
    return y0, y0 + WIDTH                              # -62 .. -52 um


def via_box() -> Box:
    """The TopVia2 square: the top ``WIDTH`` of the inner strip extension."""
    xi0, xi1 = inner_x()
    pts = spiral_centreline()
    y_end = float(pts[-1][1])                          # -12 um
    return (xi0, xi1, y_end - WIDTH, y_end, Z_VIA, Z_M2)


def underpass_box() -> Box:
    xi0, xi1 = inner_x()
    pts = spiral_centreline()
    y_end = float(pts[-1][1])
    y0, _ = port_line_y()
    return (xi0, xi1, y0, y_end, Z_M1, Z_M1 + T_M1)


def columns() -> dict[str, Box]:
    xo0, xo1 = outer_x()
    xi0, xi1 = inner_x()
    y0, y1 = port_line_y()
    return {"outer": (xo0, xo1, y0, y1, PORT_GAP, Z_M2),
            "inner": (xi0, xi1, y0, y1, PORT_GAP, Z_M1)}


def port_gaps() -> dict[str, Box]:
    xo0, xo1 = outer_x()
    xi0, xi1 = inner_x()
    y0, y1 = port_line_y()
    return {"outer": (xo0, xo1, y0, y1, 0.0, PORT_GAP),
            "inner": (xi0, xi1, y0, y1, 0.0, PORT_GAP)}


def post_x() -> tuple[float, float]:
    xi1 = inner_x()[1]
    xo0 = outer_x()[0]
    return xi1 + SHORT_GAP, xo0 - SHORT_GAP            # 22 .. 32 um


def bridge_boxes(grounded: bool = True) -> list[Box]:
    """The short standard's physical bridge as the FDFD builds it: an M2 arm
    from the outer column to the post, an M1 arm from the post to the inner
    column, and the post between them.

    ``grounded=True`` is the FDFD's own object: the post spans the WHOLE height
    from the ground plane to the M2 top, so it touches z = 0 and joins its own
    ground image -- a second real-to-image current path.  ``grounded=False``
    truncates that post to the metal stack (``Z_M1 .. Z_M2_TOP``, the M1
    bottom to the M2 top): the same M1 -> M2 connection, no ground contact.
    The Greenhouse referee's bridge is two bars meeting in the middle with NO
    vertical member and no ground contact at all, so it is the UNGROUNDED
    bridge that is the referee's quantity bar for bar (the vertical member is
    metal the referee omits, exactly as it omits the via); the grounded one is
    what the FDFD's short standard physically is."""
    xo0, xo1 = outer_x()
    xi0, xi1 = inner_x()
    y0, y1 = port_line_y()
    p0, p1 = post_x()
    z_post = 0.0 if grounded else Z_M1
    return [(p0, xo1, y0, y1, Z_M2, Z_M2_TOP),         # M2 arm 22 .. 52
            (xi0, p1, y0, y1, Z_M1, Z_M1 + T_M1),      # M1 arm 2 .. 32
            (p0, p1, y0, y1, z_post, Z_M2_TOP)]        # post (ground or M1) .. M2 top


def clip_boxes(boxes: Iterable[Box], axis: int, lo: float | None = None,
               hi: float | None = None) -> list[Box]:
    out = []
    for b in boxes:
        b = list(b)
        i0, i1 = 2 * axis, 2 * axis + 1
        if lo is not None:
            b[i0] = max(b[i0], lo)
        if hi is not None:
            b[i1] = min(b[i1], hi)
        if b[i1] - b[i0] > 1e-15:
            out.append(tuple(b))    # type: ignore[arg-type]
    return out


def mirror_boxes(boxes: Iterable[Box]) -> list[Box]:
    """The PEC-ground image: every box mirrored in ``z = 0``."""
    return [(b[0], b[1], b[2], b[3], -b[5], -b[4]) for b in boxes]


def bar_boxes(length: float, width: float = WIDTH, thickness: float = T_M2,
              z_centre: float = GROUND_H) -> list[Box]:
    return [(-0.5 * length, 0.5 * length, -0.5 * width, 0.5 * width,
             z_centre - 0.5 * thickness, z_centre + 0.5 * thickness)]


def bend_boxes(l1: float = BEND_L1, l2: float = BEND_L2, width: float = WIDTH,
               thickness: float = T_M2, z_centre: float = GROUND_H) -> list[Box]:
    """D2c's L-bend between its two reference planes: a leg of length ``l1`` in
    ``+y`` from ``(l2/2, -l1/2)`` to the corner ``(l2/2, l1/2)``, then ``l2`` in
    ``-x`` to ``(-l2/2, l1/2)`` -- the area-exact tiling of the drawn polygon."""
    hw = 0.5 * width
    z0, z1 = z_centre - 0.5 * thickness, z_centre + 0.5 * thickness
    return [(0.5 * l2 - hw, 0.5 * l2 + hw, -0.5 * l1, 0.5 * l1 + hw, z0, z1),
            (-0.5 * l2, 0.5 * l2 - hw, 0.5 * l1 - hw, 0.5 * l1 + hw, z0, z1)]


# ----------------------------------------------------------------------------
# 2. the two conductor models
#
# (a) SEGMENT model -- what a designer writes by hand: one FastHenry segment
#     per centreline piece, ``nwinc x nhinc`` filaments inside it, segments
#     meeting at the centreline corner POINT.  At low frequency the filaments
#     of a segment carry equal current densities (they share both end nodes and
#     the resistive division is uniform), so this model IS the Greenhouse
#     uniform-current bar model with FastHenry's own corner convention, and it
#     is used here to gate FastHenry against the closed forms (F1) and to run
#     the nwinc/nhinc ladders (F2).
#
# (b) MESH model -- a 3-D PEEC discretisation of the metal VOLUME: the region
#     is cut by a rectilinear grid, every cell carries a node at its centre and
#     every internal face a node at its centre, and each cell-centre-to-face
#     pair is one FastHenry segment of exactly that face's cross-section and
#     half the cell's length.  For a uniform flow along any axis the segments
#     tile the metal exactly once, so a straight run costs nothing in accuracy
#     (splitting a uniform bar lengthwise leaves the partial inductance
#     unchanged); where the current turns -- corners, the via, the transition
#     out of a lead column, the post -- the cells let it redistribute, which
#     the segment model cannot.  This is the model that answers F3/F4/F5.

def axis_lines(edges: Sequence[float], fine: float, coarse: float, fine_max: float) -> np.ndarray:
    """Grid lines on one axis: every mandatory ``edge`` is a line; an interval
    no longer than ``fine_max`` (a conductor WIDTH or slab thickness) is cut
    into cells of at most ``fine``, a longer one (a run LENGTH or a gap) into
    cells of at most ``coarse``."""
    e = np.unique(np.round(np.asarray(edges, float), 15))
    e = e[np.concatenate(([True], np.diff(e) > 1e-11))]   # never a hair-thin cell
    out = [float(e[0])]
    for a, b in zip(e[:-1], e[1:]):
        d = float(b - a)
        h = fine if d <= fine_max * (1.0 + 1e-9) else coarse
        n = max(1, int(math.ceil(d / h - 1e-9)))
        out.extend(float(a + d * (k + 1) / n) for k in range(n))
    return np.array(out, dtype=np.float64)


class Mesh:
    """Nodes and segments in metres; emitted to a FastHenry ``.inp`` in um."""

    def __init__(self) -> None:
        self.nodes: dict[str, tuple[float, float, float]] = {}
        self.segs: list[tuple[str, str, str, float, float, tuple[int, int, int]]] = []
        self.groups: dict[str, list[str]] = {}
        self._n = 0
        self._e = 0

    def node(self, xyz: tuple[float, float, float]) -> str:
        self._n += 1
        name = f"N{self._n}"
        self.nodes[name] = xyz
        return name

    def seg(self, a: str, b: str, w: float, h: float, wdir: tuple[int, int, int]) -> str:
        self._e += 1
        name = f"E{self._e}"
        self.segs.append((name, a, b, w, h, wdir))
        return name

    def add_to(self, group: str, node: str) -> None:
        self.groups.setdefault(group, []).append(node)

    def anchor(self, group: str) -> str:
        return self.groups[group][0]

    def counts(self) -> dict[str, int]:
        return {"nodes": len(self.nodes), "segments": len(self.segs),
                "groups": len(self.groups)}


WDIR = {0: (0, 1, 0), 1: (1, 0, 0), 2: (1, 0, 0)}    # width direction per segment axis


def mesh_boxes(boxes: Sequence[Box], h_fine: float, h_coarse: float, fine_max: float,
               h_z_fine: float, h_z_coarse: float, z_fine_max: float,
               cuts: Sequence[tuple[int, float, Box, str, str]] = (),
               terminals: Sequence[tuple[str, int, float, int, Box]] = ()) -> Mesh:
    """Mesh the union of ``boxes`` (see the model note above).

    ``cuts`` are internal source planes: ``(axis, coordinate, footprint, group
    below, group above)`` -- a face on that plane inside the footprint gets TWO
    nodes instead of one, so a port can be inserted there.  ``terminals`` are
    open ends: ``(group, axis, coordinate, side, footprint)`` with ``side``
    ``-1`` for the low-coordinate face of a cell and ``+1`` for the high one;
    each boundary face there gets a node and the half segment that reaches it,
    so the metal is covered right up to the terminal plane."""
    # grid lines come from the BOX edges only: a cut or terminal footprint is a
    # selector, not a geometry (its plane and its in-plane extent must already be
    # box edges, which is how every fixture below is written).  Adding footprint
    # edges here would let a footprint padded by a hair create a hair-thin cell,
    # and FastHenry bails on the degenerate filaments that follow.
    edges = [[b[2 * a] for b in boxes] + [b[2 * a + 1] for b in boxes] for a in range(3)]
    lines = [axis_lines(edges[0], h_fine, h_coarse, fine_max),
             axis_lines(edges[1], h_fine, h_coarse, fine_max),
             axis_lines(edges[2], h_z_fine, h_z_coarse, z_fine_max)]
    xs, ys, zs = lines
    cx, cy, cz = (0.5 * (v[:-1] + v[1:]) for v in lines)
    metal = np.zeros((len(cx), len(cy), len(cz)), dtype=bool)
    for b in boxes:
        ix = (cx > b[0]) & (cx < b[1])
        iy = (cy > b[2]) & (cy < b[3])
        iz = (cz > b[4]) & (cz < b[5])
        metal |= ix[:, None, None] & iy[None, :, None] & iz[None, None, :]

    m = Mesh()
    idx = np.argwhere(metal)
    centre: dict[tuple[int, int, int], str] = {}
    for i, j, k in idx:
        centre[(int(i), int(j), int(k))] = m.node((float(cx[i]), float(cy[j]), float(cz[k])))
    size = [np.diff(xs), np.diff(ys), np.diff(zs)]

    def face_geometry(axis: int, i: int, j: int, k: int,
                      side: int = 1) -> tuple[float, float, tuple[float, float, float]]:
        """(w, h, face centre) of the ``side`` face of cell (i, j, k) normal to
        ``axis`` (``side = +1``: the high-coordinate face)."""
        c = [float(cx[i]), float(cy[j]), float(cz[k])]
        c[axis] += 0.5 * side * float(size[axis][(i, j, k)[axis]])
        if axis == 0:
            return float(size[1][j]), float(size[2][k]), tuple(c)     # type: ignore[return-value]
        if axis == 1:
            return float(size[0][i]), float(size[2][k]), tuple(c)     # type: ignore[return-value]
        return float(size[0][i]), float(size[1][j]), tuple(c)         # type: ignore[return-value]

    def in_box(p: Sequence[float], b: Box) -> bool:
        return all(b[2 * a] - 1e-12 <= p[a] <= b[2 * a + 1] + 1e-12 for a in range(3))

    tol = 1e-12
    for (i, j, k), na in centre.items():
        for axis in range(3):
            nb_idx = (i + (axis == 0), j + (axis == 1), k + (axis == 2))
            w, h, fc = face_geometry(axis, i, j, k)
            if nb_idx in centre:
                cut = next((c for c in cuts if c[0] == axis and abs(fc[axis] - c[1]) < tol
                            and in_box(fc, c[2])), None)
                if cut is None:
                    nf = m.node(fc)
                    m.seg(na, nf, w, h, WDIR[axis])
                    m.seg(nf, centre[nb_idx], w, h, WDIR[axis])
                else:
                    n_lo, n_hi = m.node(fc), m.node(fc)
                    m.seg(na, n_lo, w, h, WDIR[axis])
                    m.seg(n_hi, centre[nb_idx], w, h, WDIR[axis])
                    m.add_to(cut[3], n_lo)
                    m.add_to(cut[4], n_hi)
    for group, axis, coord, side, fp in terminals:
        for (i, j, k), na in centre.items():
            w, h, fc = face_geometry(axis, i, j, k, side)
            step = 1 if side > 0 else -1
            nb_idx = (i + step * (axis == 0), j + step * (axis == 1), k + step * (axis == 2))
            if nb_idx in centre or abs(fc[axis] - coord) > tol or not in_box(fc, fp):
                continue
            nf = m.node(fc)
            m.seg(na, nf, w, h, WDIR[axis])
            m.add_to(group, nf)
    for group, _, _, _, _ in terminals:
        if not m.groups.get(group):
            raise ValueError(f"terminal {group!r} caught no face: check its plane and footprint")
    for _, _, _, g_lo, g_hi in cuts:
        if not m.groups.get(g_lo) or not m.groups.get(g_hi):
            raise ValueError(f"cut {g_lo}/{g_hi} caught no face")
    return m


def segment_chain(m: Mesh, points: Sequence[Sequence[float]], width: float, thickness: float,
                  divide: int = 1, group_start: str | None = None,
                  group_end: str | None = None, first_node: str | None = None) -> tuple[str, str]:
    """The SEGMENT model of a centreline polyline: one FastHenry segment per
    piece (``divide`` pieces per centreline piece), meeting at the corner
    points.  Returns the first and last node names."""
    pts = [np.asarray(p, float) for p in points]
    first = last = first_node
    prev: str | None = first_node
    for a, b in zip(pts[:-1], pts[1:]):
        d = b - a
        axis = int(np.argmax(np.abs(d)))
        w, h = (width, thickness) if axis != 2 else (width, width)
        for s in range(divide):
            p0 = a + d * (s / divide)
            p1 = a + d * ((s + 1) / divide)
            n0 = prev if prev is not None else m.node(tuple(float(v) for v in p0))
            n1 = m.node(tuple(float(v) for v in p1))
            m.seg(n0, n1, w, h, WDIR[axis])
            if first is None:
                first = n0
            prev = n1
    last = prev
    assert first is not None and last is not None
    if group_start:
        m.add_to(group_start, first)
    if group_end:
        m.add_to(group_end, last)
    return first, last


def mirror_mesh(m: Mesh) -> Mesh:
    """A copy of ``m`` mirrored in ``z = 0`` (the PEC-ground image).  The width
    directions are horizontal for every segment used here, so mirroring is just
    ``z -> -z``; a port driven with ``-I`` then carries exactly the image
    current (horizontal currents reverse, vertical ones do not)."""
    out = Mesh()
    ren = {}
    for name, (x, y, z) in m.nodes.items():
        ren[name] = out.node((x, y, -z))
    for _, a, b, w, h, wd in m.segs:
        out.seg(ren[a], ren[b], w, h, wd)
    for g, ns in m.groups.items():
        for n in ns:
            out.add_to(g, ren[n])
    return out


def merge(a: Mesh, b: Mesh, prefix_a: str = "a", prefix_b: str = "b") -> Mesh:
    """Two meshes side by side in one deck, group names prefixed."""
    out = Mesh()
    for src, pre in ((a, prefix_a), (b, prefix_b)):
        ren = {}
        for name, xyz in src.nodes.items():
            ren[name] = out.node(xyz)
        for _, na, nb, w, h, wd in src.segs:
            out.seg(ren[na], ren[nb], w, h, wd)
        for g, ns in src.groups.items():
            for n in ns:
                out.add_to(f"{pre}_{g}", ren[n])
    return out


# ----------------------------------------------------------------------------
# 3. the FastHenry deck, the runner and what is read back

UM = 1e6


def write_deck(mesh: Mesh, ports: Sequence[tuple[str, str, str]], freqs: tuple[float, float, float],
               sigma: float = SIGMA, nwinc: int = 1, nhinc: int = 1, title: str = "") -> str:
    """A complete ``.inp`` deck in um.  ``ports`` are ``(name, group, group)``;
    every group with more than one node is tied by ``.equiv`` (an equipotential
    terminal face, which is what a de-embedding reference plane is).
    ``freqs`` is ``(fmin, fmax, ndec)``.  ``sigma`` is SI (S/m) and is
    converted to FastHenry's per-unit-length units."""
    out = [f"* {title}", ".units um",
           f".default sigma={sigma / UM:.10g} nwinc={nwinc} nhinc={nhinc}"]
    for name, (x, y, z) in mesh.nodes.items():
        out.append(f"{name} x={x * UM:.9f} y={y * UM:.9f} z={z * UM:.9f}")
    for name, a, b, w, h, wd in mesh.segs:
        out.append(f"{name} {a} {b} w={w * UM:.9f} h={h * UM:.9f} "
                   f"wx={wd[0]} wy={wd[1]} wz={wd[2]}")
    for g, ns in mesh.groups.items():
        if len(ns) > 1:
            for i in range(0, len(ns), 30):     # keep lines short
                chunk = ns[i:i + 30]
                out.append(".equiv " + " ".join([ns[0]] + [c for c in chunk if c != ns[0]]))
    for name, ga, gb in ports:
        out.append(f".external {mesh.anchor(ga)} {mesh.anchor(gb)} {name}")
    fmin, fmax, ndec = freqs
    out.append(f".freq fmin={fmin:.10g} fmax={fmax:.10g} ndec={ndec:.10g}")
    out.append(".end")
    return "\n".join(out) + "\n"


def work_dir() -> pathlib.Path:
    cache = pathlib.Path(os.environ.get("RFX_REFEREE_CACHE",
                                       pathlib.Path.home() / ".cache" / "rfx-referees"))
    d = cache / "fasthenry" / "work"
    d.mkdir(parents=True, exist_ok=True)
    return d


def parse_zc(text: str) -> tuple[list[str], dict[float, np.ndarray]]:
    """``(port names in row order, {frequency: complex Z matrix})``."""
    names: dict[int, str] = {}
    mats: dict[float, np.ndarray] = {}
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        ln = lines[i].strip()
        if ln.startswith("Row "):
            head, _, rest = ln.partition(":")
            k = int(head.split()[1])
            names[k - 1] = rest.split("port name:")[-1].strip() if "port name:" in rest else f"p{k}"
        elif ln.startswith("Impedance matrix") or ln.startswith("ADMITTANCE"):
            freq = float(ln.split("frequency =")[1].split()[0])
            n = int(ln.split()[-3])
            rows = []
            j = i + 1
            while len(rows) < n and j < len(lines):
                vals = lines[j].replace("j", "").split()
                if len(vals) >= 2 * n:
                    rows.append([float(vals[2 * c]) + 1j * float(vals[2 * c + 1]) for c in range(n)])
                j += 1
            mats[freq] = np.array(rows, dtype=complex)
            i = j - 1
        i += 1
    order = [names[k] for k in sorted(names)]
    return order, mats


_BINARY_ID: dict[str, str] = {}
# every deck this process used, {cache key: (tag, seconds, executed?)} -- the
# study's own cost record (the cache directory may hold decks from other runs)
RUNS: dict[str, tuple[str, float, bool]] = {}


def binary_identity() -> str:
    """The SHA-256 of the FastHenry binary itself.  It goes into the key of
    every cached run, so a record written by a DIFFERENT build (an unpatched
    one, a different commit, another compiler) is never replayed as if this
    build had produced it -- without it the "live" gates of
    ``tests/unit/fdfd/test_fdfd_fasthenry_referee.py`` could pass from a stale
    cache without executing FastHenry at all."""
    b = fasthenry_binary()
    if b is None:
        raise RuntimeError("no fasthenry binary: run validation/referees/fasthenry/build_fasthenry.sh")
    key = str(b)
    if key not in _BINARY_ID:
        h = hashlib.sha256()
        with open(b, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        _BINARY_ID[key] = h.hexdigest()
    return _BINARY_ID[key]


def run_key(deck: str) -> str:
    """The cache key of one deck: the deck itself, the solver options AND the
    binary's SHA-256.  The last term is what makes a cache hit a statement
    about THIS build (see :func:`binary_identity`)."""
    return hashlib.sha1((deck + " ".join(FH_OPTS) + " "
                         + binary_identity()).encode()).hexdigest()[:16]


def run_fasthenry(deck: str, tag: str = "", fresh: bool = False) -> dict[str, Any]:
    """Run the pinned FastHenry on ``deck`` and return the parsed impedance
    matrices.  Runs are cached by the SHA-1 of the deck, the solver options AND
    the SHA-256 of the binary (see :func:`binary_identity`), so an interrupted
    study loses nothing, a re-run with the same build is free, and a re-run
    with any other build re-measures everything."""
    binary = fasthenry_binary()
    if binary is None:
        raise RuntimeError("no fasthenry binary: run validation/referees/fasthenry/build_fasthenry.sh")
    bid = binary_identity()
    key = run_key(deck)
    d = work_dir() / key
    zc = d / "Zc.mat"
    if fresh and d.exists():
        shutil.rmtree(d)
    # a run counts as cached only once it has written its own wall clock:
    # FastHenry writes Zc.mat frequency by frequency, so a killed or
    # interrupted run leaves a PARTIAL Zc.mat behind
    if not (d / "seconds").is_file():
        shutil.rmtree(d, ignore_errors=True)
    executed = False
    if not zc.is_file():
        # run in a private directory and RENAME it into place: two processes
        # (this study and the test file, say) may ask for the same deck at the
        # same time, and a half-written cache entry must never be readable
        tmp = work_dir() / f"tmp-{key}-{os.getpid()}"
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True)
        (tmp / "run.inp").write_text(deck)
        (tmp / "binary_sha256").write_text(bid + "\n")
        t0 = time.time()
        proc = subprocess.run([str(binary), *FH_OPTS, "run.inp"], cwd=tmp,
                              capture_output=True, text=True)
        (tmp / "run.log").write_text(proc.stdout + proc.stderr)
        (tmp / "seconds").write_text(f"{time.time() - t0:.6f}")
        if proc.returncode != 0 or not (tmp / "Zc.mat").is_file():
            raise RuntimeError(f"fasthenry failed ({tag}): {proc.stdout[-2000:]}{proc.stderr[-2000:]}")
        executed = True
        try:
            os.rename(tmp, d)
        except OSError:                      # another process got there first
            if not zc.is_file():
                raise
            shutil.rmtree(tmp, ignore_errors=True)
    order, mats = parse_zc(zc.read_text())
    log = (d / "run.log").read_text()
    nfil = nseg = nmesh = 0
    for ln in log.splitlines():
        if "Number of filaments:" in ln:
            nfil = int(ln.split()[-1])
        elif "Number of segments:" in ln:
            nseg = int(ln.split()[-1])
        elif "Number of meshes:" in ln:
            nmesh = int(ln.split()[-1])
    secs = float((d / "seconds").read_text())
    RUNS[key] = (tag, secs, executed or RUNS.get(key, ("", 0.0, False))[2])
    return {"ports": order, "z": mats, "dir": str(d), "tag": tag,
            "seconds": secs,
            "n_filaments": nfil, "n_segments": nseg, "n_meshes": nmesh}


def cost_block(previous: dict[str, Any] | None = None) -> dict[str, Any]:
    """What this study cost in FastHenry, from the runner's own record: every
    deck the assembly asked for, its tag and its recorded wall clock, whether
    it was executed in this process or replayed from the cache.  (The cache
    DIRECTORY holds more than this -- decks from earlier ladders, from the test
    file and from other studies -- so it is the runner's record, not the
    directory, that is counted.)  A resumed run merges its record with the one
    already in the JSON, because the blocks it reuses were measured by decks it
    never asks for again."""
    per: dict[str, Any] = dict(((previous or {}).get("per_deck") or {}))
    for key, (tag, secs, _ex) in RUNS.items():
        per[key] = [tag, round(secs, 6)]
    ex = [v for v in RUNS.values() if v[2]]
    return {"what": "FastHenry decks this study's assembly used, with the recorded wall clock of "
                    "each (tag, seconds); merged across resumed runs",
            "decks": len(per), "seconds": float(sum(v[1] for v in per.values())),
            "decks_executed_here": len(ex),
            "seconds_executed_here": float(sum(v[1] for v in ex)),
            "binary_sha256": binary_identity(),
            "cache": str(work_dir()),
            "solver_options": " ".join(FH_OPTS),
            "per_deck": per}


def l_image(res: dict[str, Any], freq: float, real: str = "preal", image: str = "pimag") -> float:
    """``Im(Z11 - Z12)/omega`` for the ``(+I, -I)`` real/image excitation: the
    partial inductance of the real conductor over an infinite PEC plane."""
    z = res["z"][freq]
    i, j = res["ports"].index(real), res["ports"].index(image)
    return float((z[i, i] - z[i, j]).imag / (2.0 * math.pi * freq))


def l_port(res: dict[str, Any], freq: float, port: str) -> float:
    z = res["z"][freq]
    i = res["ports"].index(port)
    return float(z[i, i].imag / (2.0 * math.pi * freq))


def r_port(res: dict[str, Any], freq: float, port: str) -> float:
    z = res["z"][freq]
    i = res["ports"].index(port)
    return float(z[i, i].real)


def m_ports(res: dict[str, Any], freq: float, a: str, b: str) -> float:
    z = res["z"][freq]
    i, j = res["ports"].index(a), res["ports"].index(b)
    return float(z[i, j].imag / (2.0 * math.pi * freq))


# ----------------------------------------------------------------------------
# 4. the fixtures, exactly the ones the FDFD solves
#
# "open" cases are the conductor BETWEEN THE TWO REFERENCE PLANES that the
# open/short de-embedding leaves (the referee's quantity): the real conductor
# carries port 1, its ground image port 2, and L = Im(Z11 - Z12)/omega is the
# partial inductance over the infinite PEC plane.
# "loop" cases are the WHOLE fixture -- lumped port gap, lead column, DUT,
# lead column, port gap -- whose real and image halves meet at z = 0 and close
# one loop; the source sits in the outer port gap's z = 0 plane and the
# differential inductance the FDFD reports is HALF the loop's (the fields of
# the two halves are mirror images, so the loop stores twice the half-space
# energy).

def y_reference() -> float:
    """The de-embedding reference plane of the strip: the lead-column footprint
    CENTRE line, ``y = -a0 - lead + W/2`` = -57 um."""
    return port_line_y()[0] + REF_SHIFT


def x_references() -> tuple[float, float]:
    """The same plane seen by the short standard's bridge: ``x = 47 / 7 um``."""
    return outer_x()[1] - REF_SHIFT, inner_x()[0] + REF_SHIFT


def case(name: str) -> dict[str, Any]:
    """Boxes, terminals, cut plane and port topology of one fixture."""
    y_ref = y_reference()
    xo, xi = x_references()
    py0, py1 = port_line_y()
    xo0, xo1 = outer_x()
    xi0, xi1 = inner_x()
    m2_face = (xo0, xo1, y_ref, y_ref, Z_M2, Z_M2_TOP)
    m1_face = (xi0, xi1, y_ref, y_ref, Z_M1, Z_M1 + T_M1)
    if name == "strip":
        boxes = (clip_boxes(spiral_boxes(), 1, lo=y_ref)
                 + clip_boxes([via_box()], 1, lo=y_ref)
                 + clip_boxes([underpass_box()], 1, lo=y_ref))
        return {"kind": "open", "boxes": boxes,
                "terminals": [("a", 1, y_ref, -1, m2_face), ("b", 1, y_ref, -1, m1_face)],
                "what": "M2 spiral + via + M1 underpass between the two reference planes"}
    if name == "spiral_only":
        pts = spiral_centreline()
        y_end = float(pts[-1][1])
        boxes = clip_boxes(spiral_boxes(), 1, lo=y_ref)
        return {"kind": "open", "boxes": boxes,
                "terminals": [("a", 1, y_ref, -1, m2_face),
                              ("b", 1, y_end, +1, (xi0, xi1, y_end, y_end, Z_M2, Z_M2_TOP))],
                "what": "the M2 spiral ALONE, from the outer reference plane to the inner end of "
                        "the strip (no via, no underpass): the corners in situ"}
    if name in ("bridge", "bridge_ungrounded"):
        # "bridge" is the FDFD's own object, post and all: it TOUCHES the
        # ground plane, so it carries a second real-to-image path and is
        # shorted by it.  "bridge_ungrounded" truncates that post to the metal
        # stack, which is the like-for-like counterpart of the referee's two
        # post-less bars -- see bridge_boxes().
        m2a, m1a, post = bridge_boxes(grounded=(name == "bridge"))
        boxes = clip_boxes([m2a], 0, hi=xo) + clip_boxes([m1a], 0, lo=xi) + [post]
        return {"kind": "open", "boxes": boxes,
                "terminals": [("a", 0, xo, +1, (xo, xo, py0, py1, Z_M2, Z_M2_TOP)),
                              ("b", 0, xi, -1, (xi, xi, py0, py1, Z_M1, Z_M1 + T_M1))],
                "what": ("the short standard's physical bridge (M2 arm, GROUNDED post, M1 arm) "
                         "between the same two planes -- what the FDFD's short standard is"
                         if name == "bridge" else
                         "the same bridge with the post truncated to the metal stack (M1 bottom "
                         "to M2 top, no ground contact): the referee's post-less bridge, bar for "
                         "bar -- what the referee's 333.055 pH subtracts")}
    if name in ("dut", "short"):
        gaps, cols = port_gaps(), columns()
        boxes = [gaps["outer"], gaps["inner"], cols["outer"], cols["inner"]]
        if name == "dut":
            boxes += spiral_boxes() + [via_box(), underpass_box()]
        else:
            boxes += bridge_boxes()
        # The DUT is a two-terminal element in magnetoquasistatics: with no
        # displacement current, whatever enters the outer port leaves the inner
        # one, so its 2-port Z does not exist (Y is singular) and the ONLY mode
        # is the differential one.  Shorting the inner crossing (real to image)
        # realises exactly that mode with one port: V2 = 0 and I2 = -I1 by KCL,
        # so Z(1-port) = Z11 - Z12 - Z21 + Z22 = 2 x the half-space differential
        # impedance.  The SHORT standard's grounded post is a SECOND path
        # between the real structure and its image, so its inner port current is
        # NOT forced to -I: it gets both cuts and the honest two-port
        # differential combination (which is what the FDFD's
        # Z11 - Z12 - Z21 + Z22 is, and what makes the post short itself out of
        # the answer only as far as physics says it does).
        cuts = [(2, 0.0, gaps["outer"], "p1_lo", "p1_hi")]
        if name == "short":
            cuts.append((2, 0.0, gaps["inner"], "p2_lo", "p2_hi"))
        return {"kind": "loop", "boxes": boxes, "cuts": cuts,
                "what": f"the whole {name} fixture (port gaps, lead columns and the "
                        f"{'spiral' if name == 'dut' else 'short standard'}) over the PEC ground"}
    if name == "bar":
        boxes = bar_boxes(BAR_LENGTH)
        return {"kind": "open", "boxes": boxes,
                "terminals": [("a", 0, -0.5 * BAR_LENGTH, -1, boxes[0]),
                              ("b", 0, +0.5 * BAR_LENGTH, +1, boxes[0])],
                "what": "D2c's straight bar between its reference planes"}
    if name == "bend":
        boxes = bend_boxes()
        return {"kind": "open", "boxes": boxes,
                "terminals": [("a", 1, -0.5 * BEND_L1, -1, boxes[0]),
                              ("b", 0, -0.5 * BEND_L2, -1, boxes[1])],
                "what": "D2c's L-bend between its reference planes"}
    if name in ("via_probe", "via_probe_flat"):
        # the strip's inner end, isolated: an M2 bar up the inner line, the via,
        # and the M1 underpass back down -- the ONE place where the Greenhouse
        # model has two antiparallel bars 4 um apart and the metal has a via.
        # "flat" is the same length of M2 bar with NO via and NO underpass.
        pts = spiral_centreline()
        y_end = float(pts[-1][1])
        y_start = float(pts[-2][1])
        vb = via_box()
        boxes = [(xi0, xi1, y_start, y_end, Z_M2, Z_M2_TOP)]
        term_b: tuple[str, int, float, int, Box]
        if name == "via_probe":
            boxes += [vb, (xi0, xi1, y_ref, y_end, Z_M1, Z_M1 + T_M1)]
            term_b = ("b", 1, y_ref, -1, (xi0, xi1, y_ref, y_ref, Z_M1, Z_M1 + T_M1))
        else:
            term_b = ("b", 1, y_end, +1, (xi0, xi1, y_end, y_end, Z_M2, Z_M2_TOP))
        return {"kind": "open", "boxes": boxes,
                "terminals": [("a", 1, y_start, -1, (xi0, xi1, y_start, y_start, Z_M2, Z_M2_TOP)),
                              term_b],
                "what": "the inner end of the strip alone: M2 extension, via, underpass"
                        if name == "via_probe" else "the same M2 extension alone (no via)"}
    raise ValueError(f"unknown case {name!r}")


def build_case(name: str, k: int, hl: float = WIDTH, kz: int = 1) -> tuple[Mesh, list[tuple[str, str, str]], dict[str, Any]]:
    """Mesh one fixture and its ground image and wire up the ports.  ``k`` is
    the number of cells across a conductor WIDTH (and across each metal slab
    ``kz``); ``hl`` is the largest cell along a run."""
    c = case(name)
    boxes = list(c["boxes"])
    full = boxes + mirror_boxes(boxes)
    if c["kind"] == "open":
        terms = []
        for g, ax, co, sd, fp in c["terminals"]:
            terms.append((g + "_r", ax, co, sd, fp))
            terms.append((g + "_i", ax, co, sd, (fp[0], fp[1], fp[2], fp[3], -fp[5], -fp[4])))
        m = mesh_boxes(full, WIDTH / k, hl, WIDTH, T_M2 / kz, 0.5 * PORT_GAP, T_M2,
                       terminals=terms)
        ports = [("preal", "a_r", "b_r"), ("pimag", "a_i", "b_i")]
    else:
        m = mesh_boxes(full, WIDTH / k, hl, WIDTH, T_M2 / kz, 0.5 * PORT_GAP, T_M2,
                       cuts=c["cuts"])
        ports = [(f"pdiff{i + 1}", cut[3], cut[4]) for i, cut in enumerate(c["cuts"])]
    info = {"case": name, "kind": c["kind"], "what": c["what"], "cells_across_width": k,
            "h_long_um": hl * UM, "cells_across_slab": kz, **m.counts()}
    return m, ports, info


def measure(name: str, k: int, hl: float = WIDTH, kz: int = 1, freq: float = FREQ_LOW,
            sigma: float = SIGMA, nwinc: int = 1, nhinc: int = 1,
            freqs: tuple[float, float, float] | None = None) -> dict[str, Any]:
    """Run one fixture and return ``{L at each frequency, counts, seconds}``."""
    m, ports, info = build_case(name, k, hl, kz)
    fr = freqs or (freq, freq, 1.0)
    deck = write_deck(m, ports, fr, sigma=sigma, nwinc=nwinc, nhinc=nhinc,
                      title=f"{name} k={k} hl={hl * UM:g}um kz={kz}")
    res = run_fasthenry(deck, f"{name}-k{k}")
    out = {**info, "sigma": sigma, "nwinc": nwinc, "nhinc": nhinc,
           "n_filaments": res["n_filaments"], "n_meshes": res["n_meshes"],
           "seconds": res["seconds"], "dir": res["dir"], "L": {}, "R": {},
           "n_ports": len(ports)}
    for f in sorted(res["z"]):
        if info["kind"] == "loop":
            z = res["z"][f]
            zd = z[0, 0] if len(ports) == 1 else (z[0, 0] - z[0, 1] - z[1, 0] + z[1, 1])
            out["L"][f"{f:g}"] = float(zd.imag / (2.0 * math.pi * f)) / 2.0
            out["R"][f"{f:g}"] = float(zd.real) / 2.0
        else:
            out["L"][f"{f:g}"] = l_image(res, f)
            out["R"][f"{f:g}"] = r_port(res, f, "preal")
    out["L_at"] = out["L"][f"{freq:g}"] if f"{freq:g}" in out["L"] else None
    return out


# ----------------------------------------------------------------------------
# 5. the SEGMENT model: closed-form gates (F1) and the nwinc/nhinc ladders (F2)

def with_image(m: Mesh, ports: Sequence[tuple[str, str]]) -> tuple[Mesh, list[tuple[str, str, str]]]:
    """``m`` beside its ``z -> -z`` image, one port on each.  Driven ``(+I, -I)``
    the image carries exactly the PEC-plane image current, so
    ``Im(Z11 - Z12)/omega`` is the partial inductance over the plane."""
    both = merge(m, mirror_mesh(m), "r", "i")
    out = []
    for i, (ga, gb) in enumerate(ports):
        out.append((f"preal{i or ''}", f"r_{ga}", f"r_{gb}"))
        out.append((f"pimag{i or ''}", f"i_{ga}", f"i_{gb}"))
    return both, out


def segment_bar(length: float, width: float = WIDTH, thickness: float = T_M2,
                z: float = 0.0, y: float = 0.0, divide: int = 1,
                group: str = "a") -> Mesh:
    m = Mesh()
    segment_chain(m, [(-0.5 * length, y, z), (0.5 * length, y, z)], width, thickness,
                  divide=divide, group_start=group, group_end=group + "end")
    return m


def segment_strip(divide: int = 1, with_via: bool = True) -> Mesh:
    """The SEGMENT model of the de-embedded strip: the centreline of
    ``rect_spiral`` from the outer reference plane, the via as one z-directed
    segment from the M2 centre line to the M1 centre line, and the underpass
    back to the inner plane.  ``divide`` splits every centreline piece into
    equal segments (the subdivision-along-length knob)."""
    pts = spiral_centreline()
    y_ref = y_reference()
    z2 = Z_M2 + 0.5 * T_M2
    z1 = Z_M1 + 0.5 * T_M1
    path = [(float(pts[0][0]), y_ref, z2)] + [(float(p[0]), float(p[1]), z2) for p in pts[1:]]
    m = Mesh()
    segment_chain(m, path, WIDTH, T_M2, divide=divide, group_start="a")
    end = m.segs[-1][2]
    x_in, y_in = float(pts[-1][0]), float(pts[-1][1])
    start = end
    if with_via:
        start = m.node((x_in, y_in, z1))
        m.seg(end, start, WIDTH, WIDTH, WDIR[2])
    segment_chain(m, [(x_in, y_in, z1), (x_in, y_ref, z1)], WIDTH, T_M1, divide=divide,
                  first_node=start, group_end="b")
    return m


def bridge_segment_model(divide: int = 1, split: float = 0.5) -> Mesh:
    """The SEGMENT model of the short standard's bridge, the referee's own
    construction: an M2 bar from the outer plane to the M2/M1 transition and an
    M1 bar from there to the inner plane (the post itself carries no series
    current in this model -- that is exactly the referee's assumption, and the
    MESH model is what tests it)."""
    xo, xi = x_references()
    y_b = y_reference()
    xs = xi + split * (xo - xi)
    m = Mesh()
    segment_chain(m, [(xo, y_b, Z_M2 + 0.5 * T_M2), (xs, y_b, Z_M2 + 0.5 * T_M2)],
                  WIDTH, T_M2, divide=divide, group_start="a")
    end = m.segs[-1][2]
    nv = m.node((xs, y_b, Z_M1 + 0.5 * T_M1))
    m.seg(end, nv, WIDTH, WIDTH, WDIR[2])
    segment_chain(m, [(xs, y_b, Z_M1 + 0.5 * T_M1), (xi, y_b, Z_M1 + 0.5 * T_M1)],
                  WIDTH, T_M1, divide=divide, first_node=nv, group_end="b")
    return m


def measure_mesh(m: Mesh, ports: Sequence[tuple[str, str, str]], freqs: tuple[float, float, float],
                 sigma: float = SIGMA, nwinc: int = 1, nhinc: int = 1, tag: str = "",
                 kind: str = "image") -> dict[str, Any]:
    deck = write_deck(m, ports, freqs, sigma=sigma, nwinc=nwinc, nhinc=nhinc, title=tag)
    res = run_fasthenry(deck, tag)
    out: dict[str, Any] = {"tag": tag, "nwinc": nwinc, "nhinc": nhinc, "sigma": sigma,
                           **m.counts(), "n_filaments": res["n_filaments"],
                           "n_meshes": res["n_meshes"], "seconds": res["seconds"],
                           "L": {}, "R": {}, "M": {}}
    for f in sorted(res["z"]):
        key = f"{f:g}"
        if kind == "image":
            out["L"][key] = l_image(res, f)
        elif kind == "plain":
            out["L"][key] = l_port(res, f, res["ports"][0])
        elif kind == "pair":
            out["L"][key] = l_port(res, f, res["ports"][0])
            out["M"][key] = m_ports(res, f, res["ports"][0], res["ports"][1])
        out["R"][key] = r_port(res, f, res["ports"][0])
    return out


# ----------------------------------------------------------------------------
# 6. the Greenhouse referee, recomputed here from its own module

def area_exact(sg: Any, segs: Sequence[Any]) -> list:
    """The area-exact corner convention (``spiral_convergence.area_exact_chain``,
    reimplemented): translate every bar by ``+W/2`` along its own direction
    except the start of the first and the end of the last, so the corner
    squares are covered exactly once."""
    out = []
    n = len(segs)
    for i, s in enumerate(segs):
        a = np.asarray(s.start, float)
        b = np.asarray(s.end, float)
        d = b - a
        ell = float(np.linalg.norm(d))
        u = d / ell
        sa = a if i == 0 else a + 0.5 * s.width * u
        sb = b if i == n - 1 else b + 0.5 * s.width * u
        out.append(sg.Segment(tuple(sa), tuple(sb), s.width, s.thickness, s.current))
    return out


def greenhouse_block(sg: Any) -> dict[str, Any]:
    """Every Greenhouse number this study is judged against, recomputed from
    ``spiral_greenhouse`` with the construction of
    ``invariant_ladder.referee_block`` (the de-embedded value must come back
    equal to the 333.055 pH that study published -- gate F0)."""
    pts = spiral_centreline()
    y_ref = y_reference()
    xo, xi = x_references()
    path = [(float(pts[0][0]), y_ref)] + [(float(p[0]), float(p[1])) for p in pts[1:]]
    bars = [sg.Segment((a[0], a[1], 0.0), (b[0], b[1], 0.0), WIDTH, T_M2)
            for a, b in zip(path[:-1], path[1:])]
    x_in, y_end = float(pts[-1][0]), float(pts[-1][1])
    under = sg.Segment((x_in, y_end, -DZ_UNDER), (x_in, y_ref, -DZ_UNDER), WIDTH, T_M1)
    strip_ae = sg.greenhouse_terms(area_exact(sg, bars) + [under], ground_height=GROUND_H)
    strip_cl = sg.greenhouse_terms(bars + [under], ground_height=GROUND_H)
    xs = 0.5 * (xo + xi)
    bridge = sg.greenhouse_terms(
        [sg.Segment((xo, y_ref, 0.0), (xs, y_ref, 0.0), WIDTH, T_M2),
         sg.Segment((xs, y_ref, -DZ_UNDER), (xi, y_ref, -DZ_UNDER), WIDTH, T_M1)],
        ground_height=GROUND_H)
    # the via as the segment model has it: a W x W bar from the M2 centre line to
    # the M1 centre line.  Every mutual with an x/y bar vanishes, so all it adds
    # is its own partial self-inductance and the mutual with its ground image
    # (a vertical image current flows the SAME way, so that mutual is positive).
    l_via = sg.self_inductance_bar(DZ_UNDER, WIDTH, WIDTH)
    z_c = Z_M2 + 0.5 * T_M2 - 0.5 * DZ_UNDER
    m_via = sg.mutual_inductance_bars(DZ_UNDER, WIDTH, WIDTH, DZ_UNDER, WIDTH, WIDTH,
                                      0.0, 0.0, -2.0 * z_c)
    # D2c's bar and bend, area-exact, same ground image
    bar = sg.greenhouse_terms([sg.Segment((0.5 * BAR_LENGTH, 0.0, 0.0),
                                          (-0.5 * BAR_LENGTH, 0.0, 0.0), WIDTH, T_M2)],
                              ground_height=GROUND_H)
    hw = 0.5 * WIDTH
    bend = sg.greenhouse_terms(
        [sg.Segment((0.5 * BEND_L2, -0.5 * BEND_L1, 0.0), (0.5 * BEND_L2, 0.5 * BEND_L1 + hw, 0.0),
                    WIDTH, T_M2),
         sg.Segment((0.5 * BEND_L2 - hw, 0.5 * BEND_L1, 0.0), (-0.5 * BEND_L2, 0.5 * BEND_L1, 0.0),
                    WIDTH, T_M2)], ground_height=GROUND_H)
    # the strip's inner end alone (the via probe's referee): the M2 extension and
    # the underpass as two antiparallel bars DZ_UNDER apart, no via
    y_start = float(pts[-2][1])
    probe = sg.greenhouse_terms(
        [sg.Segment((x_in, y_start, 0.0), (x_in, y_end, 0.0), WIDTH, T_M2),
         sg.Segment((x_in, y_end, -DZ_UNDER), (x_in, y_ref, -DZ_UNDER), WIDTH, T_M1)],
        ground_height=GROUND_H)
    spiral_only = sg.greenhouse_terms(area_exact(sg, bars), ground_height=GROUND_H)
    probe_flat = sg.greenhouse_terms(
        [sg.Segment((x_in, y_start, 0.0), (x_in, y_end, 0.0), WIDTH, T_M2)],
        ground_height=GROUND_H)
    out = {
        "what": "area-exact Greenhouse + PEC ground image, recomputed here from "
                "validation/crossval/comparators/spiral_greenhouse.py",
        "strip": float(strip_ae.total), "strip_centreline_convention": float(strip_cl.total),
        "bridge": float(bridge.total),
        "deembedded": float(strip_ae.total - bridge.total),
        "published_deembedded": GREENHOUSE_DEEMBEDDED,
        "matches_published": abs(float(strip_ae.total - bridge.total) / GREENHOUSE_DEEMBEDDED - 1.0),
        "strip_worst_error": float(strip_ae.worst_error),
        "via_self": float(l_via), "via_image_mutual": float(m_via),
        "segment_model_reference": float(strip_cl.total + l_via + m_via),
        "spiral_only": float(spiral_only.total),
        "bar": float(bar.total), "bend": float(bend.total),
        "corner_excess": float(bend.total - bar.total),
        "via_probe": float(probe.total), "via_probe_flat": float(probe_flat.total),
        "ground_height": GROUND_H, "reference_planes": {"y": y_ref, "x_outer": xo, "x_inner": xi},
    }
    return out


# ----------------------------------------------------------------------------
# 7. ladders, Richardson and the gates

def richardson(h: Sequence[float], values: Sequence[float]) -> dict[str, Any]:
    """Limit estimates from a refinement ladder: the finest pair at p = 1 and
    p = 2 and, when three points allow it, at the OBSERVED order solved from
    the consistent three-point equation (the ladders are NOT uniform in h, so
    the constant-ratio log formula does not apply).  The reported
    ``range`` is the span of those estimates and ``uncertainty`` half its
    width: a spread of ESTIMATORS (p = 1, p = 2, measured order), not a bound
    on the discretisation error -- see ``uncertainty_is`` in the result and the
    ``confirmation`` block, which measures one level finer and re-extrapolates
    to check the spread was honest."""
    h = [float(v) for v in h]
    f = [float(v) for v in values]
    est: dict[str, float] = {}
    if len(f) >= 2:
        for p in (1.0, 2.0):
            r = (h[-2] / h[-1]) ** p
            est[f"p{int(p)}_finest_pair"] = f[-1] + (f[-1] - f[-2]) / (r - 1.0)
    order = None
    if len(f) >= 3:
        # The observed order is solved from the CONSISTENT three-point equation
        #     (h1^p - h2^p) / (h2^p - h3^p) = (f2 - f1) / (f3 - f2),
        # by bisection, exactly as validation/fdfd/spiral_convergence.richardson
        # does. The closed form log(ratio)/log(h1/h2) that stood here is only
        # valid when the refinement ratio is CONSTANT, and none of this study's
        # ladders is uniform (the spiral goes k = 1, 2, 3, 4, 6, 8, so the ratio
        # moves 1.5 -> 1.333; the bend ends 12, 16, 24). It reported 2.237 for
        # the spiral strip and 1.11 for the bend where the data support 1.55 and
        # 1.93; the limits move by at most 0.017 pH and no gate changes.
        h1, h2, h3 = h[-3], h[-2], h[-1]
        d1, d2 = f[-2] - f[-3], f[-1] - f[-2]
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
                order = 0.5 * (lo + hi)
                est["observed_order"] = f[-1] + d2 * h3 ** order / (h2 ** order - h3 ** order)
    vals = list(est.values())
    return {"h": h, "values": f, "observed_order": order, "estimates": est,
            "range": [min(vals), max(vals)] if vals else None,
            "limit": float(np.mean([min(vals), max(vals)])) if vals else None,
            "uncertainty": 0.5 * (max(vals) - min(vals)) if vals else None,
            "uncertainty_is": "half the SPREAD of the p = 1, p = 2 and observed-order estimates "
                              "on the finest pair -- a disagreement between estimators, NOT a "
                              "bound on the discretisation error. Where the observed order is "
                              "near 2 the p = 1 member is the pessimistic one and the interval is "
                              "conservative; where it is near 1 the p = 2 member is. The check "
                              "that it is honest is the finer off-ladder level (see "
                              "\"confirmation\")"}


def rel(a: float, b: float) -> float:
    return a / b - 1.0


def union_volume(boxes: Sequence[Box]) -> float:
    """Exact volume of the union of axis-aligned boxes (the elementary-cell
    decomposition on the boxes' own edge grid)."""
    lines = [np.unique([b[2 * a] for b in boxes] + [b[2 * a + 1] for b in boxes]) for a in range(3)]
    c = [0.5 * (v[:-1] + v[1:]) for v in lines]
    d = [np.diff(v) for v in lines]
    inside = np.zeros((len(c[0]), len(c[1]), len(c[2])), dtype=bool)
    for b in boxes:
        ix = (c[0] > b[0]) & (c[0] < b[1])
        iy = (c[1] > b[2]) & (c[1] < b[3])
        iz = (c[2] > b[4]) & (c[2] < b[5])
        inside |= ix[:, None, None] & iy[None, :, None] & iz[None, None, :]
    vol = d[0][:, None, None] * d[1][None, :, None] * d[2][None, None, :]
    return float(vol[inside].sum())


def bbox(boxes: Sequence[Box]) -> list[float]:
    a = np.asarray(boxes, float)
    return [float(a[:, 0].min()), float(a[:, 1].max()), float(a[:, 2].min()),
            float(a[:, 3].max()), float(a[:, 4].min()), float(a[:, 5].max())]


# ----------------------------------------------------------------------------
# 8. the study blocks

def block_geometry(sg: Any) -> dict[str, Any]:
    """F0 (structural): the conductors this referee meshes ARE the ones the FDFD
    builds.  The centreline comes from ``rfx.fdfd.gds`` and is checked against
    the independent copy in ``spiral_greenhouse``; the metal volumes and the
    object boxes are checked against the ones the FDFD study read back from its
    BUILT cell masks (``invariant_ladder.json``, ``p1.coordinates.1``)."""
    pts = spiral_centreline()
    ref = sg.rect_spiral_centreline(N_TURNS, R_OUT, WIDTH, SPACING, LEAD)
    dev = float(np.abs(np.asarray(pts, float) - np.asarray(ref, float)).max())
    strip = spiral_boxes()
    cols = columns()
    gaps = port_gaps()
    dut = strip + [via_box(), underpass_box(), cols["outer"], cols["inner"]]
    short = bridge_boxes() + [cols["outer"], cols["inner"]]
    vols = {"dut": union_volume(dut), "open": union_volume([cols["outer"], cols["inner"]]),
            "short": union_volume(short)}
    boxes = {"dut_m2": bbox(strip), "dut_m1": list(underpass_box()), "dut_via": list(via_box()),
             "column_outer": list(cols["outer"]), "column_inner": list(cols["inner"]),
             "port_outer": list(gaps["outer"]), "port_inner": list(gaps["inner"]),
             "post": list(bridge_boxes()[2]), "bridge_m2": list(bridge_boxes()[0]),
             "bridge_m1": list(bridge_boxes()[1])}
    fdfd: dict[str, Any] = {}
    worst_vol = worst_box = None
    if INVARIANT_LADDER_PATH.is_file():
        j = json.loads(INVARIANT_LADDER_PATH.read_text())
        c1 = j["p1"]["coordinates"]["1"]
        fdfd_vol = c1["volume"]
        fdfd_box = {"dut_m2": c1["dut_m2"], "dut_m1": c1["dut_m1"], "dut_via": c1["dut_via"],
                    "column_outer": c1["columns"]["outer"], "column_inner": c1["columns"]["inner"],
                    "port_outer": c1["ports"]["outer"], "port_inner": c1["ports"]["inner"],
                    "post": c1["post"], "bridge_m2": c1["bridge_m2"], "bridge_m1": c1["bridge_m1"]}
        worst_vol = max(abs(vols[k] / fdfd_vol[k] - 1.0) for k in vols)
        worst_box = max(max(abs(np.asarray(boxes[k], float) - np.asarray(fdfd_box[k], float)))
                        for k in fdfd_box)
        fdfd = {"volume": fdfd_vol, "boxes": fdfd_box, "source": str(INVARIANT_LADDER_PATH)}
    gh = greenhouse_block(sg)
    return {"what": "the meshed conductors are the FDFD's own objects",
            "centreline_max_deviation_m": dev,
            "volumes_m3": vols, "boxes_m": boxes, "fdfd": fdfd,
            "worst_volume_rel": worst_vol, "worst_box_deviation_m": float(worst_box) if worst_box is not None else None,
            "greenhouse": gh,
            "passed": bool(dev <= 1e-15 and (worst_vol is None or worst_vol <= 1e-12)
                           and (worst_box is None or worst_box <= 1e-12)
                           and gh["matches_published"] <= 1e-12)}


def block_f1(sg: Any) -> dict[str, Any]:
    """F1: FastHenry against the Hoer-Love closed forms -- one bar, pairs of
    parallel bars, and (the ground model this referee uses) one bar over the PEC
    plane by images, each on a filament ladder."""
    out: dict[str, Any] = {"what": "closed forms: self, mutual and the image construction",
                           "frequency": FREQ_LOW, "self": {}, "mutual": {}, "image": {}}
    bars = {"strip_100um": (100e-6, WIDTH, T_M2), "cell_10um": (10e-6, WIDTH, T_M2),
            "via_4um": (DZ_UNDER, WIDTH, WIDTH)}
    for name, (ell, w, t) in bars.items():
        rows = []
        for nw, nh in ((1, 1), (2, 2), (4, 4), (8, 4)):
            m = Mesh()
            segment_chain(m, [(0.0, 0.0, 0.0), (ell, 0.0, 0.0)], w, t,
                          group_start="a", group_end="b")
            r = measure_mesh(m, [("p", "a", "b")], (FREQ_LOW, FREQ_LOW, 1.0), nwinc=nw, nhinc=nh,
                             tag=f"self-{name}-{nw}x{nh}", kind="plain")
            ref = sg.self_inductance_bar(ell, w, t)
            rows.append({"nwinc": nw, "nhinc": nh, "n_filaments": r["n_filaments"],
                         "L": r["L"][f"{FREQ_LOW:g}"], "reference": float(ref),
                         "rel": rel(r["L"][f"{FREQ_LOW:g}"], float(ref)), "seconds": r["seconds"]})
        out["self"][name] = {"length": ell, "width": w, "thickness": t, "ladder": rows,
                             "worst_rel": max(abs(x["rel"]) for x in rows)}
    pairs = {
        "coplanar_gap_W": dict(dw=2.0 * WIDTH, dt=0.0, dl=0.0),
        "stacked_underpass": dict(dw=0.0, dt=DZ_UNDER, dl=0.0),
        "collinear_gap_50um": dict(dw=0.0, dt=0.0, dl=150e-6),
        "far_10W": dict(dw=10.0 * WIDTH, dt=0.0, dl=0.0),
    }
    ell = 100e-6
    for name, off in pairs.items():
        rows = []
        for nw, nh in ((1, 1), (4, 4)):
            m = Mesh()
            segment_chain(m, [(0.0, 0.0, 0.0), (ell, 0.0, 0.0)], WIDTH, T_M2,
                          group_start="a", group_end="b")
            segment_chain(m, [(off["dl"], off["dw"], off["dt"]),
                              (off["dl"] + ell, off["dw"], off["dt"])], WIDTH, T_M2,
                          group_start="c", group_end="d")
            r = measure_mesh(m, [("p1", "a", "b"), ("p2", "c", "d")], (FREQ_LOW, FREQ_LOW, 1.0),
                             nwinc=nw, nhinc=nh, tag=f"pair-{name}-{nw}x{nh}", kind="pair")
            ref = sg.mutual_inductance_bars(ell, WIDTH, T_M2, ell, WIDTH, T_M2,
                                            off["dw"], off["dt"], off["dl"])
            got = r["M"][f"{FREQ_LOW:g}"]
            rows.append({"nwinc": nw, "nhinc": nh, "M": got, "reference": float(ref),
                         "rel": rel(got, float(ref)), "seconds": r["seconds"]})
        out["mutual"][name] = {**off, "length": ell, "ladder": rows,
                               "worst_rel": max(abs(x["rel"]) for x in rows)}
    rows = []
    for nw, nh in ((1, 1), (4, 4), (8, 4)):
        m = segment_bar(100e-6, z=GROUND_H)
        both, ports = with_image(m, [("a", "aend")])
        r = measure_mesh(both, ports, (FREQ_LOW, FREQ_LOW, 1.0), nwinc=nw, nhinc=nh,
                         tag=f"image-bar-{nw}x{nh}")
        ref = sg.greenhouse_terms([sg.Segment((-50e-6, 0.0, 0.0), (50e-6, 0.0, 0.0), WIDTH, T_M2)],
                                  ground_height=GROUND_H).total
        rows.append({"nwinc": nw, "nhinc": nh, "L": r["L"][f"{FREQ_LOW:g}"], "reference": float(ref),
                     "rel": rel(r["L"][f"{FREQ_LOW:g}"], float(ref)), "seconds": r["seconds"]})
    out["image"] = {"what": "a 100 x 10 x 2 um bar 26 um over the PEC plane: the image "
                            "construction (L = Im(Z11 - Z12)/omega) against the Greenhouse "
                            "image sum",
                    "height": GROUND_H, "ladder": rows,
                    "worst_rel": max(abs(x["rel"]) for x in rows)}
    out["worst_rel"] = max(max(v["worst_rel"] for v in out["self"].values()),
                           max(v["worst_rel"] for v in out["mutual"].values()),
                           out["image"]["worst_rel"])
    return out


MESH_KS = (1, 2, 3, 4, 6, 8)       # cells across the strip width, the mesh ladder
LADDER_KS_FREQ = 4                 # up to this level every run sweeps 1 kHz .. 100 MHz
CORNER_KS = (1, 2, 4, 6, 8, 12, 16, 24)   # the bar/bend ladder (F5), cheap fixtures
# the four fixtures the de-embedded quantities are built from, plus the
# ungrounded bridge that is the referee's own quantity (bridge_boxes())
MESH_CASES = ("strip", "bridge", "bridge_ungrounded", "dut", "short")
# one level FINER than the ladder, on the fixtures that can afford it: the
# 6-level limit is then checked against the re-extrapolation that includes it
# (gate F2), which is what turns "the +- is a spread of estimators" into a
# measurement instead of a caveat
CONFIRM_K = 10
CONFIRM_CASES = ("strip", "bridge", "bridge_ungrounded")


def freq_spec(k: int) -> tuple[float, float, float]:
    """1 kHz .. 100 MHz in one run (one L fill, one factorisation per point) for
    the affordable levels; the finest ones are run at the low-frequency limit
    only, where the FDFD protocol's own 100 MHz point has already been shown to
    agree."""
    return (FREQ_LOW, FREQ, 1.0) if k <= LADDER_KS_FREQ else (FREQ_LOW, FREQ_LOW, 1.0)


def mesh_block(ks: Sequence[int] = MESH_KS,
               reuse: dict[str, list[dict[str, Any]]] | None = None,
               dump: Callable[[], None] | None = None) -> dict[str, list[dict[str, Any]]]:
    """The mesh ladders of the five fixtures, ONE FIXTURE AT A TIME so that a
    re-run (a new fixture, a new level, an interrupted study) only measures
    what is missing: a case already on disk at exactly these levels is kept."""
    out: dict[str, list[dict[str, Any]]] = dict(reuse or {})
    for name in MESH_CASES:
        have = out.get(name) or []
        if [r["k"] for r in have] == list(ks):
            continue
        out[name] = mesh_rows((name,), ks)[name]
        if dump:
            dump()
    return out


def mesh_rows(names: Sequence[str], ks: Sequence[int] = MESH_KS, hl: float = WIDTH,
              kz: int = 1) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for name in names:
        rows = []
        for k in ks:
            r = measure(name, k, hl=hl, kz=kz, freq=FREQ_LOW, freqs=freq_spec(k))
            rows.append({"k": k, "h_over_W": 1.0 / k, **{kk: r[kk] for kk in
                         ("segments", "nodes", "n_filaments", "n_meshes", "seconds", "L", "R")}})
        out[name] = rows
    return out


def ladder_of(rows: Sequence[dict[str, Any]], key: str = f"{FREQ_LOW:g}") -> dict[str, Any]:
    h = [r["h_over_W"] for r in rows]
    v = [r["L"][key] for r in rows]
    return richardson(h, v)


def block_f2(mesh: dict[str, list[dict[str, Any]]], gh: dict[str, Any]) -> dict[str, Any]:
    """F2: filament convergence.  Two independent knobs, because the two models
    have different ones: ``nwinc/nhinc`` (and subdivision along the length) in
    the SEGMENT model, which is exact in the uniform-current limit and therefore
    must show a flat ladder; and the cell count across the width in the MESH
    model, which is where the current distribution actually converges.  Both at
    the low-frequency limit and at the FDFD protocol's 100 MHz / 2e6 S/m."""
    seg: dict[str, Any] = {"what": "segment model: nwinc x nhinc and subdivision along length",
                           "rows": []}
    for divide, nw, nh in ((1, 1, 1), (1, 2, 2), (1, 4, 4), (1, 8, 4), (2, 1, 1), (4, 1, 1),
                           (4, 4, 4)):
        m = segment_strip(divide)
        both, ports = with_image(m, [("a", "b")])
        r = measure_mesh(both, ports, (FREQ_LOW, FREQ, 1.0), nwinc=nw, nhinc=nh,
                         tag=f"seg-strip-d{divide}-{nw}x{nh}")
        seg["rows"].append({"divide": divide, "nwinc": nw, "nhinc": nh,
                            "n_filaments": r["n_filaments"], "seconds": r["seconds"],
                            "L_low": r["L"][f"{FREQ_LOW:g}"], "L_100MHz": r["L"][f"{FREQ:g}"]})
    lows = [x["L_low"] for x in seg["rows"]]
    highs = [x["L_100MHz"] for x in seg["rows"]]
    seg["spread_low"] = (max(lows) - min(lows)) / float(np.mean(lows))
    seg["spread_100MHz"] = (max(highs) - min(highs)) / float(np.mean(highs))
    seg["L_low"] = lows[0]
    seg["L_100MHz"] = highs[0]
    seg["rel_100MHz_to_low"] = rel(highs[0], lows[0])
    seg["greenhouse_segment_reference"] = gh["segment_model_reference"]
    seg["rel_to_greenhouse_segment_reference"] = rel(lows[0], gh["segment_model_reference"])

    msh: dict[str, Any] = {"what": "mesh model: cells across the strip width", "rows": []}
    for r in mesh["strip"]:
        msh["rows"].append({"k": r["k"], "segments": r["segments"], "n_meshes": r["n_meshes"],
                            "seconds": r["seconds"], "L_low": r["L"][f"{FREQ_LOW:g}"],
                            "L_100MHz": r["L"].get(f"{FREQ:g}")})
    msh["richardson_low"] = ladder_of(mesh["strip"])
    with_high = [r for r in mesh["strip"] if f"{FREQ:g}" in r["L"]]
    msh["richardson_100MHz"] = richardson([r["h_over_W"] for r in with_high],
                                          [r["L"][f"{FREQ:g}"] for r in with_high])
    msh["rel_100MHz_to_low"] = [rel(r["L"][f"{FREQ:g}"], r["L"][f"{FREQ_LOW:g}"]) for r in with_high]
    # sensitivities at one level: the cell size along a run, the cells across the
    # metal thickness, and the filaments inside each mesh segment
    sens: dict[str, Any] = {}
    base = measure("strip", 3, freqs=freq_spec(3))
    sens["base"] = base["L"][f"{FREQ_LOW:g}"]
    for tag, kw in (("h_long_half", dict(hl=0.5 * WIDTH)), ("h_long_double", dict(hl=2.0 * WIDTH)),
                    ("slab_2_cells", dict(kz=2))):
        r = measure("strip", 3, freqs=freq_spec(3), **kw)     # type: ignore[arg-type]
        sens[tag] = {"L": r["L"][f"{FREQ_LOW:g}"], "rel": rel(r["L"][f"{FREQ_LOW:g}"], sens["base"]),
                     "segments": r["segments"], "seconds": r["seconds"]}
    # filaments INSIDE each mesh segment: one frequency only -- every extra
    # filament adds mesh loops, and at 4 filaments per segment the dense mesh
    # factorisation is minutes per frequency point for a check whose answer is
    # a single number (the low-frequency limit is where "does the mesh model
    # need filaments" is asked; the 100 MHz point is answered by the mesh
    # ladder itself).
    m3, p3, _ = build_case("strip", 3)
    r = measure_mesh(m3, p3, (FREQ_LOW, FREQ_LOW, 1.0), nwinc=2, nhinc=2, tag="strip-k3-fil2x2")
    sens["filaments_2x2"] = {"L": r["L"][f"{FREQ_LOW:g}"],
                             "rel": rel(r["L"][f"{FREQ_LOW:g}"], sens["base"]),
                             "n_filaments": r["n_filaments"], "seconds": r["seconds"]}
    freqs = {f: base["L"][f] for f in sorted(base["L"], key=float)}
    return {"segment_model": seg, "mesh_model": msh, "sensitivity": sens,
            "frequency_ladder_k3": freqs,
            "frequency_plateau_rel": rel(base["L"][f"{FREQ:g}"], base["L"][f"{FREQ_LOW:g}"])}


def block_f3(mesh: dict[str, list[dict[str, Any]]], gh: dict[str, Any]) -> dict[str, Any]:
    """F3: the de-embedded quantity, three ways, because three different things
    can be meant by it and only one of them is the referee's.

    ``planes_referee``  = L(strip) - L(bridge_ungrounded): between the two
        reference planes, with the bridge's post truncated to the metal stack.
        This is the referee's quantity BAR FOR BAR -- the Greenhouse bridge is
        two post-less bars with no ground contact -- and it is what F3 compares
        with 333.055 pH.
    ``planes_physical`` = L(strip) - L(bridge): the same, with the fixture's
        PHYSICAL bridge, whose post reaches the ground plane and joins its own
        image.  That ground contact is a second real-to-image path, not a
        modelling approximation, so it is reported as its own term
        (``post_short``) rather than charged to the referee.
    ``fixture``         = L(DUT) - L(SHORT), the whole fixture driven
        differentially, port gaps and lead columns included -- exactly what the
        FDFD's open/short de-embedding measures, so F4 compares this one.

    The two steps between them are measured here: ``post_short`` (the post's
    ground contact) and ``column_transition`` (the transition into the lead
    columns, the one term the referee cancels by symmetry and the metal does
    not)."""
    out: dict[str, Any] = {"greenhouse": gh["deembedded"], "rows": {}}
    pairs = (("planes_referee", ("strip", "bridge_ungrounded")),
             ("planes_physical", ("strip", "bridge")),
             ("fixture", ("dut", "short")))
    for label, (a, b) in pairs:
        rows = []
        for ra, rb in zip(mesh[a], mesh[b]):
            assert ra["k"] == rb["k"]
            d = {f: ra["L"][f] - rb["L"][f] for f in ra["L"] if f in rb["L"]}
            rows.append({"k": ra["k"], "h_over_W": ra["h_over_W"], "L": d,
                         f"L_{a}": ra["L"], f"L_{b}": rb["L"],
                         "seconds": ra["seconds"] + rb["seconds"]})
        r = richardson([x["h_over_W"] for x in rows], [x["L"][f"{FREQ_LOW:g}"] for x in rows])
        out["rows"][label] = {"what": {"planes_referee": "between the planes, post-less bridge "
                                                         "(the referee's quantity)",
                                       "planes_physical": "between the planes, the fixture's "
                                                          "grounded-post bridge",
                                       "fixture": "the whole fixture (what the FDFD measures)"
                                       }[label],
                              "levels": rows, "richardson": r,
                              "limit": r["limit"], "uncertainty": r["uncertainty"],
                              "rel_to_greenhouse": rel(r["limit"], gh["deembedded"]),
                              "difference": r["limit"] - gh["deembedded"]}
    pr, pp, f = (out["rows"]["planes_referee"], out["rows"]["planes_physical"],
                 out["rows"]["fixture"])
    key = f"{FREQ_LOW:g}"
    out["post_short"] = {
        "what": "L(planes, post-less bridge) - L(planes, grounded-post bridge) at the continuum "
                "limit: what the short standard's post costs by REACHING the ground plane (it "
                "joins its own image there, a second real-to-image path, so the grounded bridge "
                "has the lower inductance and subtracting it leaves MORE behind). A property of "
                "the fixture, not an error of the referee",
        "note": "the image construction JOINS the two halves at z = 0 wherever a conductor "
                "touches the plane (one shared node per cell face there, two segments incident on "
                "it: the real cell and its mirror), so the grounded post is a real parallel path "
                "and not a dead end. At the COARSEST level (k = 1) the two bridges nevertheless "
                "agree to 1e-14: with one cell across the post the two arms are a balanced "
                "bridge and the single branch through z = 0 carries no current. The branch opens "
                "as soon as the current can redistribute laterally, and the difference converges "
                "monotonically from k = 2 (0.320 / 0.380 / 0.403 / 0.420 / 0.426 pH at "
                "k = 2, 3, 4, 6, 8)",
        "abs": pr["limit"] - pp["limit"], "rel": rel(pr["limit"], pp["limit"]),
        "bridge_grounded": None, "bridge_ungrounded": None,
        "per_level": [rel(a["L"][key], b["L"][key]) for a, b in zip(pr["levels"], pp["levels"])]}
    br = ladder_of(mesh["bridge"])
    bu = ladder_of(mesh["bridge_ungrounded"])
    out["post_short"]["bridge_grounded"] = br["limit"]
    out["post_short"]["bridge_ungrounded"] = bu["limit"]
    out["post_short"]["bridge_abs"] = bu["limit"] - br["limit"]
    out["post_short"]["bridge_per_level"] = [b["L"][key] - a["L"][key]
                                             for a, b in zip(mesh["bridge"],
                                                             mesh["bridge_ungrounded"])]
    out["column_transition"] = {"what": "L(fixture) - L(planes, grounded-post bridge) at the "
                                        "continuum limit: what the de-embedding does NOT cancel "
                                        "where the current turns out of the lead column",
                                "abs": f["limit"] - pp["limit"],
                                "rel": rel(f["limit"], pp["limit"]),
                                "per_level": [rel(b["L"][key], a["L"][key])
                                              for a, b in zip(pp["levels"], f["levels"])]}
    return out


def block_f5(gh: dict[str, Any]) -> dict[str, Any]:
    """F5: the cross-checks of ``corner_convergence.py`` (study D2c) -- the
    straight bar and the L-bend of the SAME fixture, between the same reference
    planes, in FastHenry against the same Greenhouse numbers.  The straight bar
    carries a uniform current, so FastHenry must reproduce the closed form at
    every level; the bend is where the corner lives."""
    out: dict[str, Any] = {"what": "D2c's bar and bend (W = 10 um, t = 2 um, l1 = l2 = 100 um, "
                                   "ground 26 um below the M2 centre) between their reference planes",
                           "greenhouse": {"bar": gh["bar"], "bend": gh["bend"],
                                          "excess": gh["corner_excess"]},
                           "levels": []}
    for k in CORNER_KS:
        b = measure("bar", k, freqs=freq_spec(k))
        d = measure("bend", k, freqs=freq_spec(k))
        row = {"k": k, "h_over_W": 1.0 / k,
               "L_bar": b["L"][f"{FREQ_LOW:g}"], "L_bend": d["L"][f"{FREQ_LOW:g}"],
               "excess": d["L"][f"{FREQ_LOW:g}"] - b["L"][f"{FREQ_LOW:g}"],
               "rel_bar": rel(b["L"][f"{FREQ_LOW:g}"], gh["bar"]),
               "rel_bend": rel(d["L"][f"{FREQ_LOW:g}"], gh["bend"]),
               "segments": [b["segments"], d["segments"]],
               "seconds": [b["seconds"], d["seconds"]]}
        if f"{FREQ:g}" in b["L"]:
            row["L_bar_100MHz"] = b["L"][f"{FREQ:g}"]
            row["L_bend_100MHz"] = d["L"][f"{FREQ:g}"]
        out["levels"].append(row)
    h = [r["h_over_W"] for r in out["levels"]]
    out["richardson_bend"] = richardson(h, [r["L_bend"] for r in out["levels"]])
    out["richardson_excess"] = richardson(h, [r["excess"] for r in out["levels"]])
    out["bar_worst_rel"] = max(abs(r["rel_bar"]) for r in out["levels"])
    lim = out["richardson_excess"]["limit"]
    unc = out["richardson_excess"]["uncertainty"]
    out["excess_limit"] = lim
    out["excess_uncertainty"] = unc
    out["excess_ratio_to_greenhouse"] = lim / gh["corner_excess"]
    out["per_corner_discrepancy"] = lim - gh["corner_excess"]
    out["bend_limit"] = out["richardson_bend"]["limit"]
    out["bend_discrepancy"] = out["richardson_bend"]["limit"] - gh["bend"]
    out["bend_uncertainty"] = out["richardson_bend"]["uncertainty"]
    # what the FDFD measured for the same corner (READ at run time; that study
    # and its JSON are owned elsewhere, so the values are recorded as read)
    fdfd: dict[str, Any] = {}
    p = HERE / "corner_convergence.json"
    if p.is_file():
        try:
            j = json.loads(p.read_text())
            lev = j.get("excess", {}).get("levels", {})
            fdfd = {"source": str(p), "read_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "per_level": {k: {"excess_fdfd": v.get("excess_fdfd"),
                                      "excess_referee": v.get("excess_referee"),
                                      "admitted": v.get("admitted")} for k, v in lev.items()}}
        except (json.JSONDecodeError, KeyError, TypeError):
            fdfd = {"source": str(p), "error": "unreadable at run time"}
    out["fdfd_corner_study"] = fdfd
    return out


def block_attribution(study: dict[str, Any], gh: dict[str, Any]) -> dict[str, Any]:
    """Where the Greenhouse referee's error comes from, term by term, in pH of
    the de-embedded 333.055 pH.

    The quantity decomposed is ``planes_referee`` -- the one that IS the
    referee's, conductor for conductor (F3): strip minus the POST-LESS bridge.
    It splits exactly:
    ``[L(strip) - L(bridge)]_FH - [L(strip) - L(bridge)]_GH = (strip error) -
    (bridge error)``, and the strip's error splits again into the M2 spiral
    ALONE between the outer plane and the inner end of the strip (``corners in
    situ``, measured on its own ladder) and the remainder, which is everything
    the via and the underpass bring.  Two independent estimates are reported
    beside them: 8 x the isolated corner discrepancy of D2c's bend (F5) for the
    first, and the isolated inner-end probe for the second.

    Two terms are NOT the referee's error and are reported apart from it: the
    short standard's post REACHING the ground plane (a second real-to-image
    path, which the referee's bridge cannot have because it has no vertical
    member at all) and the lead columns (not part of this quantity)."""
    f3 = study["f3"]
    f5 = study["f5"]
    mesh = study["spiral_mesh"]
    total = f3["rows"]["planes_referee"]["difference"]
    strip_r = ladder_of(mesh["strip"])
    bridge_r = ladder_of(mesh["bridge_ungrounded"])
    strip_err = strip_r["limit"] - gh["strip"]
    bridge_err = bridge_r["limit"] - gh["bridge"]
    so = study.get("spiral_only") or {}
    corners_in_situ = (so.get("limit") - gh["spiral_only"]) if so else None
    via_rest = (strip_err - corners_in_situ) if corners_in_situ is not None else None
    probe = study["f2"].get("via_probe")
    terms = {
        "strip": {"what": "the conductor between the planes: M2 spiral + via + underpass",
                  "fasthenry": strip_r["limit"], "uncertainty": strip_r["uncertainty"],
                  "greenhouse": gh["strip"], "total": strip_err,
                  "rel": rel(strip_r["limit"], gh["strip"])},
        "bridge": {"what": "the POST-LESS bridge, the referee's own bridge conductor for "
                           "conductor (enters with a MINUS sign). The referee has no vertical "
                           "member where the metal has one, so this term is the referee's bridge "
                           "being LOW, not high",
                   "fasthenry": bridge_r["limit"], "uncertainty": bridge_r["uncertainty"],
                   "greenhouse": gh["bridge"], "total": -bridge_err,
                   "rel": rel(bridge_r["limit"], gh["bridge"])},
    }
    inside_strip = {
        "corners_in_situ": {"what": "the M2 spiral alone (8 right-angle corners, no via, no "
                                    "underpass) against the same area-exact Greenhouse bars",
                            "fasthenry": so.get("limit"), "uncertainty": so.get("uncertainty"),
                            "greenhouse": gh["spiral_only"], "total": corners_in_situ,
                            "rel": rel(so["limit"], gh["spiral_only"]) if so else None},
        "via_and_underpass": {"what": "the rest of the strip's error: the via, the underpass and "
                                      "their couplings (strip error - corners in situ)",
                              "total": via_rest},
        "isolated_corner_check": {"what": "8 x (FastHenry bend limit - Greenhouse bend) from D2c's "
                                          "ISOLATED corner (F5): an independent estimate of "
                                          "corners_in_situ, which it cannot match exactly because "
                                          "an isolated corner has no neighbouring turns to couple "
                                          "to",
                                  "per_corner": f5["bend_discrepancy"], "n": 8,
                                  "total": 8.0 * f5["bend_discrepancy"],
                                  "uncertainty": 8.0 * (f5["bend_uncertainty"] or 0.0)},
        "inner_end_probe_check": {"what": "the isolated inner end (M2 extension + via + underpass) "
                                          "against the referee's two antiparallel bars: an "
                                          "independent estimate of via_and_underpass",
                                  "total": (probe or {}).get("difference"),
                                  "uncertainty": (probe or {}).get("uncertainty"),
                                  "flat_control_rel": (probe or {}).get("flat_control", {}).get("rel")},
    }
    seg = study["f2"]["segment_model"]
    ps = f3["post_short"]
    return {"what": "the Greenhouse referee's error on ITS de-embedded quantity (strip minus the "
                    "post-less bridge), decomposed",
            "quantity": "planes_referee",
            "total_measured": total,
            "total_rel": rel(f3["rows"]["planes_referee"]["limit"], gh["deembedded"]),
            "terms": terms, "inside_strip": inside_strip,
            "sum_of_terms": float(terms["strip"]["total"] + terms["bridge"]["total"]),
            "closes_to": total - float(terms["strip"]["total"] + terms["bridge"]["total"]),
            "not_the_referees_error": {
                "what": "differences between the FDFD's fixture and the referee's quantity that "
                        "are physics, not referee modelling error -- measured here, and the reason "
                        "F3 reports three de-embedded quantities",
                "post_ground_contact": {"what": "the short standard's post reaches z = 0 and joins "
                                                "its own image: L(post-less bridge) - L(grounded "
                                                "bridge) at the limit, which moves the "
                                                "between-planes quantity by the same amount with "
                                                "the opposite sign",
                                        "bridge_abs": ps["bridge_abs"],
                                        "quantity_abs": ps["abs"], "quantity_rel": ps["rel"]},
                "lead_columns": {"what": "NOT part of the referee's quantity: L(whole fixture) - "
                                         "L(between the planes, physical bridge) at the limit, the "
                                         "transition out of the lead columns that the de-embedding "
                                         "cancels by symmetry and the metal does not",
                                 "total": f3["column_transition"]["abs"],
                                 "rel": f3["column_transition"]["rel"]}},
            "segment_model_check": {
                "what": "the SEGMENT model (uniform current, centreline corners, the via as one "
                        "z-directed bar) reproduces the Greenhouse sum bar for bar: the plumbing "
                        "gate on the geometry, the image construction and the ports",
                "fasthenry": seg["L_low"], "greenhouse": gh["segment_model_reference"],
                "rel": seg["rel_to_greenhouse_segment_reference"]},
            "redistribution": {
                "what": "mesh limit - segment model on the SAME conductor (the strip): everything "
                        "the uniform-current assumption gets wrong at once",
                "mesh_limit": strip_r["limit"], "segment_model": seg["L_low"],
                "abs": strip_r["limit"] - seg["L_low"],
                "rel": rel(strip_r["limit"], seg["L_low"])}}


def spiral_only_block(ks: Sequence[int] = MESH_KS) -> dict[str, Any]:
    """The M2 spiral ALONE between the outer plane and the inner end of the
    strip: the corners in situ, with no via and no underpass to confound them."""
    rows = mesh_rows(("spiral_only",), [k for k in ks if k >= 2])["spiral_only"]
    r = ladder_of(rows)
    return {"what": "the M2 spiral alone (no via, no underpass), same reference plane, same "
                    "ground image",
            "rows": [{"k": x["k"], "L": x["L"][f"{FREQ_LOW:g}"], "segments": x["segments"],
                      "seconds": x["seconds"]} for x in rows],
            "richardson": r, "limit": r["limit"], "uncertainty": r["uncertainty"]}


def confirmation_block(mesh: dict[str, list[dict[str, Any]]], k: int = CONFIRM_K) -> dict[str, Any]:
    """One level FINER than the ladder, on the three fixtures that can afford
    it, and the re-extrapolation that includes it.

    Every limit in this study is a Richardson extrapolation whose quoted +- is
    the SPREAD of the p = 1 / p = 2 / observed-order estimates -- a
    disagreement between estimators, not a bound on the discretisation error.
    The only way to find out whether that spread was honest is to measure a
    level the ladder did not use and extrapolate again from the three finest
    points: if the two limits agree inside their own spreads, the spread was
    not optimistic.  ``k = 10`` is the finest level the dense mesh solve can
    still afford for the strip (the whole DUT at this level is out of reach,
    which is why the ``fixture`` quantity is not re-extrapolated here)."""
    key = f"{FREQ_LOW:g}"
    rows: dict[str, Any] = {}
    for name in CONFIRM_CASES:
        r = measure(name, k, freqs=(FREQ_LOW, FREQ_LOW, 1.0))
        rows[name] = {"k": k, "h_over_W": 1.0 / k, "L": r["L"][key],
                      "segments": r["segments"], "seconds": r["seconds"]}
    out: dict[str, Any] = {"what": f"the strip and both bridges at k = {k}, one level finer than "
                                   "the ladder, and the re-extrapolation from the three finest "
                                   "levels including it",
                           "k": k, "levels": rows, "checks": {}}
    def series(name: str) -> list[tuple[float, float]]:
        if name == "strip":
            pts = [(r["h_over_W"], r["L"][key]) for r in mesh["strip"]]
            pts.append((rows["strip"]["h_over_W"], rows["strip"]["L"]))
        else:
            b = "bridge_ungrounded" if name == "planes_referee" else "bridge"
            pts = [(ra["h_over_W"], ra["L"][key] - rb["L"][key])
                   for ra, rb in zip(mesh["strip"], mesh[b])]
            pts.append((rows["strip"]["h_over_W"], rows["strip"]["L"] - rows[b]["L"]))
        return sorted(pts, key=lambda t: -t[0])
    for name in ("strip", "planes_referee", "planes_physical"):
        pts = series(name)
        full = richardson([h for h, _ in pts[:-1]], [v for _, v in pts[:-1]])
        three = richardson([h for h, _ in pts[-3:]], [v for _, v in pts[-3:]])
        d = three["limit"] - full["limit"]
        tol = (full["uncertainty"] or 0.0) + (three["uncertainty"] or 0.0)
        out["checks"][name] = {
            "ladder_limit": full["limit"], "ladder_uncertainty": full["uncertainty"],
            "ladder_observed_order": full["observed_order"],
            "finest_measured": pts[-1][1], "finest_h_over_W": pts[-1][0],
            "refined_limit": three["limit"], "refined_uncertainty": three["uncertainty"],
            "refined_observed_order": three["observed_order"],
            "difference": d, "tolerance": tol,
            "consistent": bool(abs(d) <= tol) if tol > 0 else None}
    out["worst_difference"] = max(abs(v["difference"]) for v in out["checks"].values())
    out["all_consistent"] = all(bool(v["consistent"]) for v in out["checks"].values())
    return out


def block_f4(study: dict[str, Any], gh: dict[str, Any]) -> dict[str, Any]:
    """F4: the V1 re-judgement.  The FDFD's Richardson range (read from
    ``invariant_ladder.json`` at run time) against this referee's continuum
    limit.

    WHICH CORRECTION APPLIES TO WHICH QUANTITY.  The FDFD's ``L_dut`` is the
    open/short (Koolen) de-embedded ``Im(Z11-Z12-Z21+Z22)/omega`` of the whole
    fixture WITH the physical grounded short standard, so its like-for-like
    counterpart is the FastHenry ``fixture`` quantity and the primary gate uses
    that one.  The only model differences between them are the ones study P
    measured and did not apply: the 10 W walls (+0.186 %) and the RC
    contamination of ``Im Z / omega`` (+0.095 %).  Study P's third correction,
    the post short (-0.33 to -0.36 %, from ITS straight-bar thru probe),
    converts ``L_dut`` into a whole-bar-subtracted third quantity; it is
    recorded here but NOT applied, because this study measures the two steps
    from ``fixture`` to the referee's quantity itself -- the lead-column
    transition and the post's ground contact (F3) -- and using both would
    double-count.  The FDFD range is therefore carried to each FastHenry
    quantity by its own measured conversion."""
    fdfd_range = list(FDFD_LIMIT_RANGE)
    src = None
    levels = None
    estimates: dict[str, Any] = {}
    corr: dict[str, Any] = {}
    f3 = study["f3"]
    col = f3["column_transition"]["rel"]        # fixture / planes_physical - 1  (> 0)
    post = f3["post_short"]["rel"]              # planes_referee / planes_physical - 1  (< 0)
    if INVARIANT_LADDER_PATH.is_file():
        j = json.loads(INVARIANT_LADDER_PATH.read_text())
        fdfd_range = [float(v) for v in j["gates"]["P2"]["range"]]
        levels = {str(k): v["L_dut"] for k, v in j["levels"].items()}
        estimates = dict((j["gates"]["P2"].get("estimates") or {}))
        src = str(INVARIANT_LADDER_PATH)
        c = j.get("corrections") or {}
        w = float((c.get("walls") or {}).get("rel") or 0.0)
        rc = float((c.get("rc") or {}).get("rel") or 0.0)
        ps = [float(v) for v in ((c.get("post_short") or {}).get("rel_range") or [0.0, 0.0])]
        fix = [v * (1.0 + w + rc) for v in fdfd_range]
        phys = [v / (1.0 + col) for v in fix]
        refq = [v * (1.0 + post) for v in phys]
        corr = {"walls_rel": w, "rc_rel": rc,
                "for_fixture": fix, "for_planes_physical": phys, "for_planes_referee": refq,
                "conversion_rel": {"column_transition": col, "post_ground_contact": post},
                "post_short_rel_range_read_not_applied": ps,
                "what": "the FDFD range moved by study P's own measured model differences (walls "
                        "and RC, which apply to every quantity here) and then carried between the "
                        "quantities by the conversions measured in F3 (the lead-column transition "
                        "and the post's ground contact). Study P's post-short correction is read "
                        "and NOT applied: it converts L_dut into a third quantity using that "
                        "study's straight-bar thru probe, and this study measures the same two "
                        "steps itself"}
    out: dict[str, Any] = {"what": "the FDFD continuum limit against the FastHenry one",
                           "fdfd_range": fdfd_range, "fdfd_source": src, "fdfd_levels": levels,
                           "fdfd_estimates": estimates, "fdfd_corrected": corr,
                           "greenhouse": gh["deembedded"], "per_quantity": {}}
    for label in ("planes_referee", "planes_physical", "fixture"):
        r = study["f3"]["rows"][label]
        lim, unc = r["limit"], r["uncertainty"]
        worst = max(abs(rel(lim, v)) for v in fdfd_range)
        best = min(abs(rel(lim, v)) for v in fdfd_range)
        inside = fdfd_range[0] - (unc or 0.0) <= lim <= fdfd_range[1] + (unc or 0.0)
        crange = corr.get(f"for_{label}") or fdfd_range
        out["per_quantity"][label] = {
            "fasthenry_limit": lim, "uncertainty": unc,
            "rel_to_fdfd_range": [rel(lim, v) for v in fdfd_range],
            "rel_to_corrected_fdfd_range": [rel(lim, v) for v in crange],
            "corrected_fdfd_range": crange,
            "worst_rel_corrected": max(abs(rel(lim, v)) for v in crange),
            "worst_rel": worst, "best_rel": best, "inside_fdfd_range": bool(inside),
            "within_5_percent": bool(worst <= 0.05), "within_3_percent": bool(worst <= 0.03),
            "rel_to_greenhouse": rel(lim, gh["deembedded"])}
    # WHAT IS EXTRAPOLATED AND WHAT IS MEASURED.  "the FDFD range contains the
    # FastHenry limit" is a statement about two extrapolations.  The FDFD's
    # finest MEASURED level and each of its three individual limit estimates
    # are reported against the FastHenry fixture limit so the reader can see
    # how much of the agreement is extrapolation: the finest measured level is
    # still several times further away than the observed-order estimate.
    fix_lim = out["per_quantity"]["fixture"]["fasthenry_limit"]
    honest: dict[str, Any] = {
        "what": "the FDFD side, un-extrapolated and estimator by estimator, against the FastHenry "
                "fixture limit -- because 'the range contains it' compares two extrapolations",
        "fasthenry_fixture_limit": fix_lim}
    if levels:
        finest = max(levels, key=lambda k: int(k))
        honest["fdfd_finest_level"] = {"m": int(finest), "L_dut": levels[finest],
                                       "rel_to_fasthenry": rel(float(levels[finest]), fix_lim)}
    if estimates:
        honest["fdfd_estimates_rel_to_fasthenry"] = {k: rel(float(v), fix_lim)
                                                     for k, v in estimates.items()}
        best = min(estimates, key=lambda k: abs(rel(float(estimates[k]), fix_lim)))
        honest["closest_estimate"] = {"which": best, "value": float(estimates[best]),
                                      "rel_to_fasthenry": rel(float(estimates[best]), fix_lim)}
    out["extrapolation_honesty"] = honest
    return out


def evaluate_gates(study: dict[str, Any]) -> dict[str, Any]:
    g: dict[str, Any] = {}
    geo = study["geometry"]
    g["F0"] = {"what": "structural: the meshed conductors are the FDFD's own objects (centreline, "
                       "metal volumes and object boxes) and the Greenhouse referee recomputed here "
                       "is the published 333.055 pH",
               "centreline_max_deviation_m": geo["centreline_max_deviation_m"],
               "worst_volume_rel": geo["worst_volume_rel"],
               "worst_box_deviation_m": geo["worst_box_deviation_m"],
               "greenhouse_matches_published": geo["greenhouse"]["matches_published"],
               "tolerance": {"deviation_m": 1e-12, "volume_rel": 1e-12},
               "passed": bool(geo["passed"])}
    f1 = study["f1"]
    g["F1"] = {"what": "closed forms: Hoer-Love self, Hoer-Love/Grover mutual and the image "
                       "construction, after the filament ladder",
               "worst_rel": f1["worst_rel"], "tolerance": 1e-3,
               "self": {k: v["worst_rel"] for k, v in f1["self"].items()},
               "mutual": {k: v["worst_rel"] for k, v in f1["mutual"].items()},
               "image": f1["image"]["worst_rel"],
               "passed": bool(f1["worst_rel"] <= 1e-3)}
    f2 = study["f2"]
    conf = study.get("confirmation") or {}
    msh = f2["mesh_model"]
    diffs = [b - a for a, b in zip([r["L_low"] for r in msh["rows"]][:-1],
                                   [r["L_low"] for r in msh["rows"]][1:])]
    contracting = all(abs(b) < abs(a) for a, b in zip(diffs[:-1], diffs[1:]))
    g["F2"] = {"what": "filament convergence: the segment model's nwinc/nhinc/subdivision ladder is "
                       "flat (the uniform-current limit is exact in it) and the mesh model's cell "
                       "ladder converges, at the low-frequency limit and at 100 MHz / 2e6 S/m alike",
               "segment_spread_low": f2["segment_model"]["spread_low"],
               "segment_spread_100MHz": f2["segment_model"]["spread_100MHz"],
               "segment_tolerance": 1e-4,
               "mesh_L_low": [r["L_low"] for r in msh["rows"]],
               "mesh_differences": diffs, "monotone": all(d < 0 for d in diffs),
               "contracting": contracting,
               "limit": msh["richardson_low"]["limit"],
               "uncertainty": msh["richardson_low"]["uncertainty"],
               "observed_order": msh["richardson_low"]["observed_order"],
               "frequency_plateau_rel": f2["frequency_plateau_rel"],
               "frequency_tolerance": 2e-3,
               # and the extrapolation itself: re-extrapolating with one level
               # FINER than the ladder (k = 10, "confirmation") must land
               # inside the two estimate spreads added together, or the quoted
               # +- was optimistic
               "confirmation_worst_difference": conf.get("worst_difference"),
               "confirmation_consistent": conf.get("all_consistent"),
               "passed": bool(f2["segment_model"]["spread_low"] <= 1e-4
                              and f2["segment_model"]["spread_100MHz"] <= 1e-4
                              and contracting
                              and abs(f2["frequency_plateau_rel"]) <= 2e-3
                              and (conf.get("all_consistent") is not False))}
    f3 = study["f3"]
    p = f3["rows"]["planes_referee"]
    g["F3"] = {"what": "the de-embedded quantity against the Greenhouse referee, on the quantity "
                       "that IS the referee's (strip minus the POST-LESS bridge): the difference "
                       "must be RESOLVED (larger than the extrapolation spread), which is the "
                       "statement that the referee has an error at all",
               "greenhouse": f3["greenhouse"],
               "fasthenry_planes_referee": p["limit"], "uncertainty": p["uncertainty"],
               "difference": p["difference"], "rel": p["rel_to_greenhouse"],
               "fasthenry_planes_physical": f3["rows"]["planes_physical"]["limit"],
               "planes_physical_rel": f3["rows"]["planes_physical"]["rel_to_greenhouse"],
               "fasthenry_fixture": f3["rows"]["fixture"]["limit"],
               "fixture_rel": f3["rows"]["fixture"]["rel_to_greenhouse"],
               "post_ground_contact_rel": f3["post_short"]["rel"],
               "column_transition_rel": f3["column_transition"]["rel"],
               "resolved": bool(abs(p["difference"]) > 3.0 * (p["uncertainty"] or math.inf)),
               "passed": bool(abs(p["difference"]) > 3.0 * (p["uncertainty"] or math.inf))}
    f4 = study["f4"]
    q = f4["per_quantity"]["fixture"]
    g["F4"] = {"what": "the V1 re-judgement: the FDFD's Richardson range against this referee's "
                       "continuum limit (the fixture quantity is the like-for-like one)",
               "fdfd_range": f4["fdfd_range"],
               "fasthenry_fixture": q["fasthenry_limit"], "uncertainty": q["uncertainty"],
               "rel_to_fdfd_range": q["rel_to_fdfd_range"], "worst_rel": q["worst_rel"],
               "rel_to_corrected_fdfd_range": q.get("rel_to_corrected_fdfd_range"),
               "worst_rel_corrected": q.get("worst_rel_corrected"),
               "inside_fdfd_range": q["inside_fdfd_range"],
               "within_5_percent": q["within_5_percent"], "within_3_percent": q["within_3_percent"],
               "planes_referee_worst_rel": f4["per_quantity"]["planes_referee"]["worst_rel"],
               "planes_physical_worst_rel": f4["per_quantity"]["planes_physical"]["worst_rel"],
               # the un-extrapolated FDFD side, so the verdict is not read as
               # an agreement of two extrapolations alone
               "fdfd_finest_level_rel": (f4.get("extrapolation_honesty") or {}
                                         ).get("fdfd_finest_level", {}).get("rel_to_fasthenry"),
               "fdfd_closest_estimate_rel": (f4.get("extrapolation_honesty") or {}
                                             ).get("closest_estimate", {}).get("rel_to_fasthenry"),
               "passed": bool(q["within_5_percent"])}
    f5 = study["f5"]
    g["F5"] = {"what": "D2c's bar and bend: the bar must reproduce the closed form at every level "
                       "(uniform current) and the bend must reproduce the corner excess's SIGN and "
                       "mechanism while measuring how much of it the referee misses",
               "bar_worst_rel": f5["bar_worst_rel"], "bar_tolerance": 1e-5,
               "greenhouse_excess": f5["greenhouse"]["excess"],
               "fasthenry_excess": f5["excess_limit"], "uncertainty": f5["excess_uncertainty"],
               "ratio": f5["excess_ratio_to_greenhouse"],
               "sign_agrees": bool(f5["excess_limit"] < 0 and f5["greenhouse"]["excess"] < 0),
               "per_corner_discrepancy": f5["per_corner_discrepancy"],
               "passed": bool(f5["bar_worst_rel"] <= 1e-5 and f5["excess_limit"] < 0
                              and f5["greenhouse"]["excess"] < 0)}
    g["all_passed"] = all(v["passed"] for v in g.values() if isinstance(v, dict))
    return g


# ----------------------------------------------------------------------------
# 9. assembly, figure, CLI

def assemble(reuse: dict[str, Any] | None = None, ks: Sequence[int] = MESH_KS,
             dump: Callable[[dict[str, Any]], None] | None = None,
             confirm_k: int = CONFIRM_K) -> dict[str, Any]:
    t0 = time.time()
    sg = load_referee()
    study: dict[str, Any] = {
        "what": "study F: FastHenry as an independent magnetoquasistatic referee for the FDFD "
                "spiral extraction and for the Greenhouse referee",
        "fasthenry": manifest(),
        "fixture": {"n_turns": N_TURNS, "r_out": R_OUT, "spacing": SPACING, "width": WIDTH,
                    "lead": LEAD, "t_m2": T_M2, "t_m1": T_M1, "t_via": T_VIA,
                    "ground_height": GROUND_H, "dz_underpass": DZ_UNDER,
                    "port_gap": PORT_GAP, "short_gap": SHORT_GAP,
                    "reference_plane_y": y_reference(), "reference_plane_x": list(x_references()),
                    "sigma": SIGMA, "freq_low": FREQ_LOW, "freq_protocol": FREQ,
                    "ground_model": "image construction: every conductor mirrored in z = 0 as a "
                                    "second port, driven (+I, -I); exact for an infinite PEC plane",
                    "solver_options": " ".join(FH_OPTS)},
    }
    if reuse:
        study.update({k: v for k, v in reuse.items() if k in
                      ("geometry", "f1", "spiral_mesh", "f2", "f3", "f4", "f5", "attribution",
                       "spiral_only", "confirmation", "cost")})

    def block(name: str, fn: Callable[[], Any]) -> None:
        if name in study and study[name]:
            return
        study[name] = fn()
        if dump:
            dump(study)

    block("geometry", lambda: block_geometry(sg))
    gh = study["geometry"]["greenhouse"]
    block("f1", lambda: block_f1(sg))
    study["spiral_mesh"] = mesh_block(ks, study.get("spiral_mesh"),
                                      (lambda: dump(study)) if dump else None)
    if "f2" not in study or not study["f2"]:
        f2 = block_f2(study["spiral_mesh"], gh)
        ks_probe = (2, 4, 6, 8)
        pr = [measure("via_probe", k, freqs=freq_spec(k))["L"][f"{FREQ_LOW:g}"] for k in ks_probe]
        pf = [measure("via_probe_flat", k, freqs=freq_spec(k))["L"][f"{FREQ_LOW:g}"]
              for k in ks_probe]
        rr = richardson([1.0 / k for k in ks_probe], pr)
        rf = richardson([1.0 / k for k in ks_probe], pf)
        f2["via_probe"] = {"what": "the strip's inner end alone: M2 extension + via + underpass "
                                   "against the referee's two antiparallel bars with no via; the "
                                   "flat control is the same M2 extension with neither",
                           "k": list(ks_probe), "L": pr, "richardson": rr,
                           "fasthenry": rr["limit"], "greenhouse": gh["via_probe"],
                           "difference": rr["limit"] - gh["via_probe"],
                           "uncertainty": rr["uncertainty"],
                           "rel": rel(rr["limit"], gh["via_probe"]),
                           "flat_control": {"L": pf, "fasthenry": rf["limit"],
                                            "greenhouse": gh["via_probe_flat"],
                                            "rel": rel(rf["limit"], gh["via_probe_flat"]),
                                            "difference": rf["limit"] - gh["via_probe_flat"]}}
        study["f2"] = f2
        if dump:
            dump(study)
    block("f3", lambda: block_f3(study["spiral_mesh"], gh))
    block("f5", lambda: block_f5(gh))
    if confirm_k:
        block("confirmation", lambda: confirmation_block(study["spiral_mesh"], confirm_k))
    block("f4", lambda: block_f4(study, gh))
    block("spiral_only", lambda: spiral_only_block(ks))
    block("attribution", lambda: block_attribution(study, gh))
    study["cost"] = cost_block(study.get("cost"))
    study["gates"] = evaluate_gates(study)
    study["seconds_assembly"] = time.time() - t0
    return study


C_BLUE, C_ORANGE, C_AQUA, C_VIOLET, C_RED = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7", "#e34948"
C_INK, C_INK2, C_MUTED, C_SURF = "#0b0b0b", "#52514e", "#a3a29c", "#fcfcfb"


def figure(study: dict[str, Any], path: pathlib.Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "axes.edgecolor": C_MUTED, "axes.labelcolor": C_INK2,
                         "xtick.color": C_INK2, "ytick.color": C_INK2, "axes.titlesize": 10,
                         "axes.titlecolor": C_INK, "axes.grid": True, "grid.color": "#e6e5e0",
                         "grid.linewidth": 0.6, "figure.facecolor": C_SURF,
                         "axes.facecolor": C_SURF, "savefig.facecolor": C_SURF})
    fig, axes = plt.subplots(2, 3, figsize=(15.0, 8.4))
    gh = study["geometry"]["greenhouse"]["deembedded"] * 1e12

    a = axes[0][0]
    f3 = study["f3"]
    fdfd = [v * 1e12 for v in study["f4"]["fdfd_range"]]
    a.axhspan(fdfd[0], fdfd[1], color=C_AQUA, alpha=0.18, lw=0)
    a.text(0.02, 0.5 * (fdfd[0] + fdfd[1]), " FDFD Richardson range", color=C_AQUA,
           va="center", fontsize=8)
    a.axhline(gh, color=C_RED, lw=1.3)
    a.text(0.02, gh + 0.4, f" Greenhouse referee {gh:.2f} pH", color=C_RED, fontsize=8)
    for label, colour, mark in (("planes_referee", C_BLUE, "o"),
                                ("planes_physical", C_ORANGE, "^"),
                                ("fixture", C_VIOLET, "s")):
        rows = f3["rows"][label]["levels"]
        h = [r["h_over_W"] for r in rows]
        v = [r["L"][f"{FREQ_LOW:g}"] * 1e12 for r in rows]
        a.plot(h, v, mark + "-", color=colour, ms=4, label=f"FastHenry, {label}")
        lim = f3["rows"][label]["limit"] * 1e12
        unc = (f3["rows"][label]["uncertainty"] or 0.0) * 1e12
        a.errorbar([0.0], [lim], yerr=[unc], fmt=mark, color=colour, ms=6, capsize=3)
    conf = (study.get("confirmation") or {}).get("checks", {})
    for label, colour, mark in (("planes_referee", C_BLUE, "o"),
                                ("planes_physical", C_ORANGE, "^")):
        c = conf.get(label)
        if c:
            a.plot([c["finest_h_over_W"]], [c["finest_measured"] * 1e12], mark, color=colour,
                   ms=7, mfc="none", mew=1.4)
    a.set_xlabel("h / W (cells across the strip width)")
    a.set_ylabel("de-embedded L (pH)")
    a.set_title("(a) the de-embedded inductance, FastHenry vs both referees")
    a.legend(fontsize=8, loc="lower right")
    a.set_xlim(-0.03, 1.05)

    a = axes[0][1]
    f1 = study["f1"]
    names, vals = [], []
    for k, v in f1["self"].items():
        names.append("self " + k)
        vals.append(abs(v["worst_rel"]))
    for k, v in f1["mutual"].items():
        names.append("M " + k)
        vals.append(abs(v["worst_rel"]))
    names.append("image bar")
    vals.append(abs(f1["image"]["worst_rel"]))
    y = np.arange(len(names))
    a.barh(y, np.maximum(vals, 1e-16), color=C_BLUE)
    a.set_yticks(y, names, fontsize=7)
    a.set_xscale("log")
    a.axvline(1e-3, color=C_RED, lw=1.2)
    a.text(1e-3, len(names) - 0.6, " gate 0.1 %", color=C_RED, fontsize=8)
    a.set_xlabel("|FastHenry / closed form - 1|")
    a.set_title("(b) F1: the Hoer-Love closed forms")

    a = axes[0][2]
    f5 = study["f5"]
    h = [r["h_over_W"] for r in f5["levels"]]
    a.plot(h, [r["excess"] * 1e12 for r in f5["levels"]], "o-", color=C_BLUE, ms=4,
           label="FastHenry excess")
    a.axhline(f5["greenhouse"]["excess"] * 1e12, color=C_RED, lw=1.3, label="Greenhouse excess")
    lim = f5["excess_limit"] * 1e12
    a.errorbar([0.0], [lim], yerr=[(f5["excess_uncertainty"] or 0.0) * 1e12], fmt="o",
               color=C_BLUE, ms=6, capsize=3)
    a.set_xlabel("h / W")
    a.set_ylabel("L(bend) - L(bar) at equal length (pH)")
    a.set_title("(c) F5: one right-angle corner")
    a.legend(fontsize=8)

    a = axes[1][0]
    fl = study["f2"]["frequency_ladder_k3"]
    fs = sorted(float(k) for k in fl)
    a.semilogx(fs, [fl[f"{f:g}"] * 1e12 for f in fs], "o-", color=C_VIOLET, ms=4)
    a.axvline(FREQ, color=C_INK2, lw=0.8, ls=":")
    a.text(FREQ, min(fl.values()) * 1e12, " FDFD protocol", color=C_INK2, fontsize=8, rotation=90,
           va="bottom")
    a.set_xlabel("frequency (Hz), sigma = 2e6 S/m")
    a.set_ylabel("L(strip) (pH)")
    a.set_title("(d) F2: the uniform-current plateau (3 cells across W)")

    a = axes[1][1]
    at = study["attribution"]
    nre = at["not_the_referees_error"]
    labels = ["corners\nin situ", "via +\nunderpass", "-(bridge)", "(post to\nground)",
              "(lead\ncolumns)"]
    vals = [at["inside_strip"]["corners_in_situ"]["total"],
            at["inside_strip"]["via_and_underpass"]["total"],
            at["terms"]["bridge"]["total"],
            nre["post_ground_contact"]["quantity_abs"], nre["lead_columns"]["total"]]
    vals = [(v or 0.0) * 1e12 for v in vals]
    a.bar(labels, vals, color=[C_BLUE, C_ORANGE, C_AQUA, C_MUTED, C_MUTED])
    a.axhline(at["total_measured"] * 1e12, color=C_RED, lw=1.3)
    a.text(0.0, at["total_measured"] * 1e12, f" measured total {at['total_measured'] * 1e12:.2f} pH",
           color=C_RED, fontsize=8, va="bottom")
    a.set_ylabel("FastHenry - Greenhouse (pH)")
    a.set_title("(e) the referee's error (grey: not the referee's, F3)")
    a.tick_params(axis="x", labelsize=8)

    a = axes[1][2]
    for name, colour in (("strip", C_BLUE), ("dut", C_VIOLET), ("short", C_ORANGE),
                         ("bridge", C_AQUA), ("bridge_ungrounded", C_MUTED)):
        rows = study["spiral_mesh"][name]
        a.loglog([r["segments"] for r in rows], [max(r["seconds"], 1e-3) for r in rows], "o-",
                 color=colour, ms=4, label=name)
    a.set_xlabel("FastHenry segments")
    a.set_ylabel("seconds (dense mesh solve)")
    a.set_title("(f) cost")
    a.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--from-json", action="store_true",
                    help="reassemble every block from the JSON on disk; build and solve nothing")
    ap.add_argument("--fresh", action="store_true", help="ignore the blocks already on disk")
    ap.add_argument("--no-figure", action="store_true")
    ap.add_argument("--ks", type=int, nargs="*", default=list(MESH_KS),
                    help="cells across the strip width for the mesh ladder")
    ap.add_argument("--confirm-k", type=int, default=CONFIRM_K,
                    help="the off-ladder confirmation level (0 skips it)")
    args = ap.parse_args(argv)

    old: dict[str, Any] = {}
    if JSON_PATH.is_file() and not args.fresh:
        old = json.loads(JSON_PATH.read_text())
    if args.from_json:
        if not old:
            print(f"--from-json needs {JSON_PATH}", file=sys.stderr)
            return 1
        study = dict(old)
        study["gates"] = evaluate_gates(study)
    else:
        if fasthenry_binary() is None:
            print("no fasthenry binary; run validation/referees/fasthenry/build_fasthenry.sh",
                  file=sys.stderr)
            return 1

        def dump(s: dict[str, Any]) -> None:
            JSON_PATH.write_text(json.dumps(s, indent=1, sort_keys=False) + "\n")

        study = assemble(old, tuple(args.ks), dump, args.confirm_k)
    JSON_PATH.write_text(json.dumps(study, indent=1, sort_keys=False) + "\n")
    if not args.no_figure:
        figure(study, PNG_PATH)
    g = study["gates"]
    for name in sorted(k for k in g if k.startswith("F")):
        print(f"{name}: {'PASS' if g[name]['passed'] else 'FAIL'}  {g[name]['what'][:70]}")
    print(f"wrote {JSON_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
