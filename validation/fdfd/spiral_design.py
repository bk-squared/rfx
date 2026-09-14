"""Design study D3: the FIRST gradient-based design loop on the 3-D FDFD
spiral inductor (``rfx.fdfd.spiral``), against a 3x3x3 parameter sweep over
the same box.

What this measures, and what it does NOT
----------------------------------------
The message of this study is SOLVE COUNT and CONSTRAINT ACCURACY, not
"gradients are free" and not absolute inductance.

* Absolute L is BIASED. The convergence study D2
  (``validation/fdfd/spiral_convergence.py``) measured this solver's
  de-embedded ``L_diff`` against an independent Greenhouse
  partial-inductance referee under a physics-consistent protocol
  (uniform-current volumetric metal, vacuum dielectrics, graded wall
  padding, area-exact corners) and found it 21.2 / 12.0 / 9.8 % LOW at
  ``base_dx`` = W/1, W/2, W/3 against the referee quantity the open/short
  fixture actually measures, L(strip) - L(short's bridge) = 333.1 pH
  (24.9 / 16.1 / 14.0 % against the strip alone), with the Richardson
  extrapolation still 5.2-7.9 % low (2.3-5.5 % after the measured
  vertical-grid correction): gate V1 FAILED. The same comparison is repeated HERE at
  this study's own nominal geometry (``referee_bias`` in the JSON) so the
  number quoted is one measured on the geometry actually optimised, not
  one carried over. A design loop on the W/1 grid therefore optimises a
  CONSISTENTLY BIASED model and no absolute-accuracy claim is made for
  ``L_target`` or for the final ``L``.
* SHAPE DERIVATIVES are the part that is validated. D2's gate V3 measured
  ``dL/dtheta`` of the FDFD against the referee's own FD4 derivative to
  within 5-15 % per component at W/1-W/3 (ratios 0.92 / 0.94 / 0.88 at
  W/3 against the de-embedded referee), and gate V2 measured ``jax.grad`` against a 4th-order central
  finite difference of the same discrete model to 6e-7. Gate O4 below
  repeats the FD4 check at the FINAL theta and on the FULL objective (not
  just L). A loop driven by derivatives that are right to 5-15 % on a
  model whose absolute value is 15-25 % low will find the right SHAPE
  change; it will not predict the absolute L of the fabricated part.
* The W/2 re-solve (gate O5) is the honest statement of what the bias does
  to the ANSWER: the final theta is re-solved on the next finer grid and
  the L and Q shifts are reported. D2's per-level gaps imply about +10 %
  on L from W/1 to W/2; no pass/fail is attached, because a shift of that
  size is exactly what a biased coarse model is expected to show.

The design problem
------------------
``f0 = 2.4 GHz``; Leontovich copper (``sigma_metal = 3e7`` S/m) on the
metal and lossy silicon (``sigma_si = 10`` S/m) with the REAL stack
permittivities (``eps_si = 11.9``, ``eps_ox = 4.1``) -- D2 used vacuum
dielectrics because its referee is magnetoquasistatic; a design study must
not. ``pad_cells = 4`` graded cells on the four side walls and the lid, so
the PEC box does not dominate (D2 measured the closed box 28 % low on L at
W/1 and the padded box converged to 0.06 %).

LEONTOVICH VALIDITY IS MARGINAL HERE and is reported, not hidden:
``spiral.leontovich_validity`` = skin depth / thinnest metal slab =
1.876 um / 2 um = 0.938 at 2.4 GHz for sigma = 3e7 S/m (measured in this
session and in the JSON as ``nominal.leontovich_validity``). The sheet
model needs that ratio WELL BELOW 1; at 0.94 the 2 um metal is one skin
depth thick, so the sheet over-estimates the internal inductance and the
loss and the ABSOLUTE Q reported here is not trustworthy either. Everything
in this study is a COMPARISON at fixed frequency, conductivity and metal
thickness -- gradient loop vs sweep, W/1 vs W/2 -- where that systematic is
common to both sides.

``theta = (r_out, spacing, width)``; ``n_turns = 2`` and the lead are
static. The box is

    r_out    64 .. 80 um     (nominal 72)
    spacing   8 .. 12 um     (nominal 10)
    width     8 .. 12 um     (nominal 10)

and every one of its corners keeps ``spiral.check_feasible`` true: the
binding constraint of a 2-turn spiral is ``a_in > width/2``, i.e. ``r_out >
3 W + 2 S``, whose worst corner here is ``(64, 12, 12) um`` with 4.0 um of
slack (the nominal has 22 um). This is why the nominal ``r_out`` is 72 um
and not the 52 um of D2: at ``r_out = 52`` with W = S = 10 um the slack is
2 um and there is no design box at all. The optimiser works on the SCALED
variables ``u = theta / theta_nominal`` (all three ~1), so the reported
gradients and the FD steps are dimensionless.

Objective::

    loss(theta) = -Q_diff(f0) / Q_ref + lam ((L_diff(f0) - L_t) / L_t)^2
    L_t = 0.85 L_diff(theta_nominal),   Q_ref = Q_diff(theta_nominal)

i.e. maximise Q at a 15 %-reduced inductance. ``lam`` is not guessed: at an
interior stationary point ``-g_Q + 2 lam e g_L = 0`` with ``e = (L - L_t) /
L_t``, so ``|e| <= |g_Q| / (2 lam |g_L|)`` and

    lam = |g_Q| / (2 e_target |g_L|),   e_target = 1 %

with ``g_Q = d(Q/Q_ref)/du`` and ``g_L = d(L/L_t)/du`` MEASURED by two
reverse-mode passes at the nominal theta (``lambda`` in the JSON). That
bound is the design of the penalty; gate O2 measures what the loop actually
achieved.

Cost model (what "one solve" means)
-----------------------------------
One objective evaluation = one FORWARD solve of the three fixtures (DUT,
OPEN, SHORT: three SuperLU factorisations of the same sparsity pattern) plus
one ADJOINT pass (three transposed solves that REUSE those factorisations
through the LU cache of ``rfx.fdfd.linear_solve``). Counted rather than
timed (``factorisations`` in the JSON): the forward solve is 3 ``_factor``
calls with 3 misses, ``value_and_grad`` is 6 calls with the SAME 3 misses,
so the adjoint adds no factorisation at all. One sweep point = one forward
solve, no adjoint.

And the honest caveat: with THREE parameters reverse mode is not the point.
Forward mode (``jax.jvp``, 3 tangents) would cost 3 extra passes per
gradient against the adjoint's 1, but those are passes over the SAME three
factorisations, and a pass that hits the cache was measured at 0.33 s
against a 38.6 s solve -- so at this size both modes are one factorisation
set plus a few percent, within about 3 % of each other. What the gradient buys here over the sweep is not
asymptotic cheapness in the parameter count, it is that 27 blind samples do
not land on the constraint surface at all, while a few dozen guided ones
land on it to 1 %. That is what gates O2 and O3 measure.

Gates and RESULTS (every number below is the one in the JSON on disk)
---------------------------------------------------------------------
The gradient loop, the sweep and the FD stencil are the recorded run:
their raw per-evaluation logs are in the JSON and the derived blocks
(gates, the lambda summary, the figure) are recomputed from those logs by
``--from-json``, never re-timed. Everything else -- the nominal, lambda,
the three memory probes, the factorisation accounting, the referee
cross-check and the W/2 re-solve -- was measured in this session, and the
nominal and lambda came back bit-identical to the recorded run
(``L_diff`` = 5.090106167675779e-10 H, ``lam`` = 11.242904519421213 to the
last digit), which is the drift check ``--verify`` does.
Fixture: W/1 grid (25, 27, 15), N = 33352 unknowns, one three-fixture
forward solve 38.6-44.0 s depending on the load of this shared machine. Nominal L_diff = 509.011 pH, Q_diff = 11.0566, so
L_target = 432.659 pH and Q_ref = 11.0566. Measured gradients at the
nominal theta: ``g_L = (+1.907, -0.338, -0.787)``, ``g_Q = (+0.288,
-0.104, +0.357)`` (scaled units), ``|g_Q| = 0.4701``, ``|g_L| = 2.0907``,
hence ``lam = 11.243``.

O1  PASS. 12 accepted iterates, loss -0.649875 -> -1.062610, worst
    increase between consecutive accepted losses -6.58e-5 (<= 0).
O2  PASS. Final ``|L - L_t| / L_t = 1.019 %`` (gate 2 %). The penalty
    weight was sized for 1 % and delivered 1.019 %: the bound is not an
    identity because ``g_Q`` and ``g_L`` are measured at the NOMINAL theta
    and because the optimum sits on two ACTIVE box bounds, so the interior
    stationarity argument only holds in the free subspace. Predictive to
    1.9 % relative, which is the point of choosing lam by measurement.
O3  NO SWEEP POINT QUALIFIES. All 27 sweep points are feasible and NONE
    lands inside the 2 % L band -- the band is 2 % wide in a quantity that
    ranges over -22 % .. +66 % across this box, and a 3x3x3 grid has no
    reason to hit it. Widening the band: 0 points at 2 %, 4 at 5 %
    (best Q = 11.7502 at (72, 10, 12) um, L +2.93 %), 7 at 10 %
    (best Q = 11.9500 at (72, 8, 12) um, L +9.02 %), 17 at 25 %. The
    gradient optimum has Q = 11.7618 at L +1.019 %: higher Q than the best
    5 %-band sweep point AND 2.9x closer to the target. On the objective
    the loop wins outright: -1.062610 against the sweep's best -1.053077.
    That comparison is the honest form of the gate and is what the test
    asserts.
O4  PASS. ``jax.grad`` vs FD4 (1 % scaled steps) at the final theta:
    grad = (8.6134e-5, 1.78423e-2, -3.99357e-1), FD4 = (8.5832e-5,
    1.78423e-2, -3.99357e-1); absolute errors (3.0e-7, 6.1e-9, 4.3e-8),
    i.e. 7.55e-7 of the gradient's sup norm (gate 1e-4). Per COMPONENT the
    relative agreements are 3.5e-3 / 3.4e-7 / 1.1e-7: the 3.5e-3 is the
    free direction, whose derivative is ZERO at a constrained optimum
    (8.6e-5 = 2.2e-4 of the norm), so a per-component relative metric
    divides the FD truncation floor by zero-ish. Both forms are recorded;
    the gate is on the norm-relative one. From the same 12 solves,
    dL/dtheta = (7.983e-10, -1.306e-10, -3.056e-10) and dQ/dtheta =
    (+4.675, -0.962, +2.626), and the chain rule ``dloss = -dQ/Q_ref +
    2 lam e dL/L_t`` reproduces the objective's FD4 to 3.55e-7.
O5  REPORT ONLY. W/2 (N = 87438, 214 s per forward solve, 18.21 GB):
      nominal theta  L 509.011 -> 543.673 pH (+6.810 %), Q 11.0566 ->
                     11.0652 (+0.077 %)
      final theta    L 437.069 -> 465.341 pH (+6.468 %), Q 11.7618 ->
                     11.6621 (-0.848 %)
    So the ANSWER moves: on the W/2 grid the optimum misses the (W/1)
    target by +7.554 % instead of +1.019 %. But the shift is almost pure
    COMMON MODE -- +6.810 % at the nominal against +6.468 % at the
    optimum, a 0.341-point differential on a 6.810-point shift, i.e.
    ``common_mode_fraction`` = 0.950 -- which is the quantitative version
    of "the derivatives are right and the absolute value is not": the
    design DIRECTION survives the grid change, the absolute target does
    not. D2's own W/1 -> W/2 move on ITS fixture is +11.65 %, quoted for
    its SIGN only: it was measured at r_out = 52 um with vacuum
    dielectrics and uniform-current metal, so nothing about its size
    transfers to this geometry, and no window is asserted around it.
BIAS (``referee_bias``, measured HERE, not carried over). At this study's
    nominal geometry, under D2's physics-consistent protocol
    (uniform-current volumetric metal, vacuum dielectrics, 100 MHz), the
    FDFD gives L_diff = 482.397 pH against the area-exact Greenhouse
    referee's 601.412 pH: -19.79 % against the strip alone, -17.6 % with
    the same 16.3 pH short-standard bridge subtracted as in D2's primary
    convention (D2 measured -21.2 % / -24.9 % on its r_out = 52 um
    fixture at the same W/1 resolution; the corner systematic here is
    +0.60 %). That is the size of the absolute error the design loop
    optimises through.
M1  FAILED, with numbers (the memory rule of this machine, section 0b and
    "Environment" below). Peak physical footprint per block against the
    limit that block ran under, in GB:

        nominal 4.53   lambda 5.63   loop 5.29 (16 chunks of ONE
        evaluation)   probe 4.96   referee_bias 4.92        limit 6.0, ok
        sweep 8.39   fd_check 21.64                         limit 6.0, OVER
        factorisations 8.38 (12)   probe_jit 15.08 (16)
        refine, the W/2 re-solve 18.21 (22)                 raised, ok
        probe_grad 8.51 (6.0)                               over BY DESIGN

    The two failures are the sweep and the FD stencil, both measured
    before this instrumentation existed and both of them several jitted
    solves in one process. The reverse-mode probe crosses the limit on
    purpose -- it is the measurement OF the crossing and stops itself
    there -- and is listed apart. The three raised limits were passed on
    the command line for one block each and are recorded per block in the
    JSON. Nothing is widened to make the gate pass: it is reported
    FAILING.

Cost, measured (``factorisations``)
-----------------------------------
Counted, not timed -- on a machine with three other agents on it the
seconds move by 40 % between minutes, so the claim is made on the LU
factorisation COUNT (``factorisations`` in the JSON: ``_factor`` is
wrapped for the duration of the block and the wrapper removed again;
nothing in ``rfx`` is modified). Measured in one process at the final
theta:

    forward           wall 38.6 s   _factor calls 3, misses 3, 38.2 s
                                    inside ``_factor`` (99.2 % of it)
    forward_cached    wall  0.33 s  calls 3, misses 0  (0.85 % of a solve)
    value_and_grad    wall 25.7 s   calls 6, misses 3, 38.0 s summed

So: one forward solve IS its three LU factorisations; the ADJOINT ADDS
ZERO of them (six calls, the same three misses -- three transposed
back-substitutions on the cached factors); and a pass that re-solves the
same point entirely from the cache costs 0.33 s, i.e. 0.85 % of a solve.
Forward mode with three tangents would add three such back-substitution
passes, ~2.5 % of a solve, which is why with three parameters reverse
mode is NOT the story here.

The story is the count. Measured, summing the per-solve seconds the
blocks logged: the loop spent 16 objective evaluations = 16 forward + 16
adjoint passes, 450.5 s, and landed on the constraint to 1.019 %; the
sweep spent 27 forward solves, 1053.6 s, and landed nowhere near it; and
the FD4 gradient at ONE point (gate O4) cost 12 forward solves, 493.1 s,
more than the entire loop.

Wall-clock warning, and the one number that looks wrong. ``value_and_grad``
comes out BELOW the forward-only solve (25.7 s against 38.6 s) while doing
strictly more work. That is not the adjoint being cheaper than the solve
it differentiates: the summed factorisation time INSIDE that 25.7 s is
38.0 s (ratio 1.48), i.e. the reverse-mode graph has enough independent
work for XLA to run the three host callbacks CONCURRENTLY (SuperLU
releases the GIL) while the forward-only graph serialises them. A
scheduling property of this backend, recorded and never used as a cost
claim. The four-execution-mode wall-time block (``mode_cost``) that would
compare forward / cached / reverse / forward mode within one process is
NOT RUN, and the JSON says why under ``not_run``: at +3.3 to +4.0 GB of
physical footprint per factorising pass (see below) it needs 20-25 GB,
which this shared 38 GB machine cannot give one of four agents.

Run::

    .venv/bin/python validation/fdfd/spiral_design.py [--no-refine]
        [--no-fd] [--no-sweep] [--maxiter 30] [--from-json]
        [--max-solves K] [--recompute BLOCK ...] [--verify]
        [--mem-limit GB] [--memory-probe K] [--memory-probe-jit K]
        [--memory-probe-grad K] [--mode-cost]

``--from-json`` is also the REASSEMBLY path: with every block already in
the JSON it re-runs nothing (no solve, no 450 s loop) and only recomputes
what is derived from the raw logs -- the gate dictionary, the seconds
summary and the figure. That is how the gates and the PNG are refreshed
after a reporting fix.

``--verify`` is the script-only drift check: it re-solves the nominal theta
on the CURRENT ``rfx.fdfd.spiral`` and compares with the JSON (one 44 s
solve, too slow for the test file). The same check was made here by
recomputing the whole ``nominal`` and ``lambda`` blocks against a
concurrently edited ``spiral.py``: ``L_diff``, ``Q_diff``, both gradients
and ``lam`` came back bit-identical to the recorded run (0.0e0 relative).

writes ``spiral_design.json`` and ``spiral_design.png`` next to this file.
The JSON is rewritten after every block AND after every objective
evaluation, so an interrupted run keeps everything already measured.
``tests/test_fdfd_spiral_design.py`` asserts the gates from that JSON and
re-runs a 2-iterate smoke test of the same driver on a cheap model
(8 tests, 56.7 s measured in this session, of which 55.4 s is that live
smoke test and 0.8 s the seven JSON gates).

Environment, and how to rerun it
--------------------------------
Two things about this machine shaped the run and are reported rather than
hidden:

1. MEMORY. An earlier revision of this file carried two contradictory
   claims (+3.6 GB of RSS per three-fixture solve in the docstring,
   ~0.5 GB in the budget comment). Both were RSS, and RSS is the wrong
   number here (section 0b). The measurement that settles it is ONE
   controlled series per execution path, all three taken in this session
   in a single process at the loop's own grid (N = 33352), same theta,
   ``clear_factor_cache()`` and ``gc.collect()`` between solves, physical
   footprint (``memory.probe``, ``memory.probe_jit``, ``memory.probe_grad``
   in the JSON):

     solves in one process         1      2      3      4      5
     solve_spiral, NO jit        4.94   4.96   4.69   4.70   4.79  GB
     the same solve JITTED       5.03   8.37  11.71  15.08         GB
     jax.value_and_grad of it    4.60   8.51                       GB

   i.e. the un-jitted call is FLAT (-0.05 GB per solve, fit residual
   0.12 GB, and the resident series agrees: 5.05 -> 4.96 GB) while the
   JITTED path -- the one the loop, the sweep and the FD stencil all take
   -- grows +3.351 GB per forward solve (fit residual 0.007 GB over four
   points, resident +2.495 GB) and the reverse pass +3.904 GB. THE single
   number this study runs on is that +3.35 GB per jitted forward solve,
   +3.90 GB per value_and_grad. It has since been diagnosed and fixed in
   ``rfx.fdfd.linear_solve`` (``validation/fdfd/memory_probe.py``): scipy
   1.18.1's ``SuperLU`` does not free its factor when it is deallocated on
   a thread other than the one that built it, and under ``jax.jit`` the
   factors were built on XLA's callback threads and dropped on the caller's.
   Every factor now lives on one dedicated thread and the jitted N = 33352
   path measures flat (5.22 / 5.33 / 5.08 / 5.02 GB over four solves). The
   numbers above are the PRE-FIX record this study ran under;
   ``--max-solves`` remains as an optional budget and is no longer needed.

   The consequences, measured and not smoothed over:

   * the gradient loop gets exactly ONE evaluation per process and
     resumes by replaying its log: the 16 recorded evaluations peak at
     4.60-5.29 GB each, all inside the 6 GB rule. The same 16 in ONE
     process would have reached about 4.60 + 15 x 3.904 = 63 GB, which is
     more than this machine has; any claim that the recorded loop ran as
     one 16-solve process would be false, and gate M1 records the 16
     per-chunk footprints that show it did not;
   * the two blocks measured BEFORE this instrumentation existed did put
     several jitted solves in one process and DID break the 6 GB rule:
     the sweep reached 8.39 GB (2 solves per process) and the FD stencil
     21.64 GB (up to 6). Gate M1 therefore FAILS, with those numbers, and
     nothing is widened to hide it: re-running those two blocks one solve
     per process costs 39 x 40 s and buys no new physics, so the honest
     record is the failing gate.
2. A ~530 s WATCHDOG on detached background processes killed two full runs
   mid-sweep (at 550 s and 529 s of wall time) with no traceback. Both
   are the reason every solve block here is RESUMABLE and the run is
   driven as short chunks, e.g.::

       until .venv/bin/python validation/fdfd/spiral_design.py \
               --from-json --max-solves 5; do :; done

   Each chunk reuses the completed blocks, continues the partial one, and
   exits 3 while work remains. L-BFGS-B holds state that cannot be
   pickled, so the gradient loop is chunked by REPLAY instead (see
   ``run_loop``): 16 chunks of ONE evaluation each, 450.5 s of solve time
   in total, the log replayed exactly (0 mismatches) at the start of each
   chunk. The claim that it ran as a single 16-solve process would be
   false, and would also be impossible: 63 GB, see above.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import pathlib
import time
from typing import Any, Callable, Sequence

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
JSON_PATH = HERE / "spiral_design.json"
PNG_PATH = HERE / "spiral_design.png"
CONV_PATH = HERE / "spiral_convergence.py"
REFEREE_PATH = REPO / "validation" / "crossval" / "comparators" / "spiral_greenhouse.py"

# ---- the design fixture ----------------------------------------------------
FREQ = 2.4e9
SIGMA_CU = 3e7             # the task's copper; delta/t = 0.938 at 2.4 GHz (marginal, see the doc)
SIGMA_SI = 10.0            # lossy substrate
N_TURNS = 2
R_OUT, SPACING, WIDTH = 72e-6, 10e-6, 10e-6
THETA0 = (R_OUT, SPACING, WIDTH)
LEAD = 2.0 * WIDTH
MARGIN = 10e-6
T_SI, T_OX_LOW, T_M1, T_VIA, T_M2, T_OX_HIGH, T_AIR = 20e-6, 1e-6, 2e-6, 2e-6, 2e-6, 3e-6, 10e-6
EPS_SI, EPS_OX = 11.9, 4.1     # the REAL stack (D2 used vacuum for its magnetoquasistatic referee)
BASE_DZ = 10e-6
METAL_CELLS = 2
PAD_CELLS, PAD_RATIO = 4, 1.5
GROUND_H = T_SI + T_OX_LOW + T_M1 + T_VIA + 0.5 * T_M2      # 26 um, ground -> M2 centre line
DZ_UNDER = 0.5 * T_M1 + T_VIA + 0.5 * T_M2                  # 4 um, M2 centre -> M1 centre
SIGMA_VOL = 3e6            # D2's uniform-current plateau, for the referee cross-check only

# ---- the design problem ----------------------------------------------------
BOUNDS = ((64e-6, 80e-6), (8e-6, 12e-6), (8e-6, 12e-6))
L_FRACTION = 0.85          # L_target = 0.85 x nominal L_diff
E_TARGET = 0.01            # the relative L error lam is sized to allow (see the doc)
L_BAND = 0.02              # the 2 % band of gates O2 / O3
FD_STEP_REL = 0.01         # FD4 step on the SCALED variables (1 % of nominal)
GTOL_FACTOR = 1e-3         # stop when max|projected grad| < GTOL_FACTOR x its nominal value
MAXITER = 30
PARAMS = ("r_out", "spacing", "width")
SWEEP_N = 3


# ----------------------------------------------------------------------------
# 0. solve budget (memory control)
#
# MEASURED in this session, one process, W/1 grid (N = 33352), same theta,
# LU factor cache emptied (``clear_factor_cache``) and ``gc.collect()``
# forced between solves -- the three ``memory_probe`` series of section 0b,
# physical footprint in GB after 1, 2, ... solves:
#
#     solve_spiral, NO jit        4.94  4.96  4.69  4.70  4.79   FLAT
#     the same solve JITTED       5.03  8.37 11.71 15.08         +3.351/solve
#     jax.value_and_grad of it    4.60  8.51                     +3.904/pass
#
# (PRE-FIX record: the growth was a scipy SuperLU cross-thread deallocation
# leak, since fixed in rfx.fdfd.linear_solve; see validation/fdfd/
# memory_probe.json, where the jitted N = 33352 path is flat.)
# The first solve of a process costs ~5 GB whatever the path; what GREW is
# the jitted one, which is the path every block of this study takes, and it
# grew linearly (fit residual 0.007 GB over the four points). At the ~6 GB
# this machine can give one of four concurrent agents that is ONE jitted
# solve, or one value_and_grad pass, per process. The study therefore runs
# its solve blocks under a BUDGET: ``--max-solves K`` stops cleanly after K
# three-fixture solves, writes the JSON, and exits with code 3 to say "more
# work to do"; every solve block resumes from what is already in the JSON,
# and the gradient loop resumes by replaying its evaluation log. Drive it
# from the shell:
#
#     until .venv/bin/python validation/fdfd/spiral_design.py \
#             --from-json --max-solves 1; do :; done

_BUDGET: dict[str, Any] = {"used": 0, "max": None}
EXIT_INCOMPLETE = 3


def spend(n: int = 1) -> None:
    _BUDGET["used"] += int(n)


def exhausted() -> bool:
    return _BUDGET["max"] is not None and _BUDGET["used"] >= int(_BUDGET["max"])


# ----------------------------------------------------------------------------
# 0b. memory accounting: WHICH number, and why not ``ru_maxrss``
#
# This machine runs four agents on 38 GB and the rule for this track is a
# peak of ~6 GB per solve process. Getting that right needed the right
# METRIC, and the obvious ones are wrong here.
#
# MEASURED on the loop run (2026-09-14), on a process that had run 12
# three-fixture ``value_and_grad`` evaluations of the W/1 grid: ru_maxrss said
# 5.93 GB and ``ps -o rss=`` said 0.007 GB, while the PHYSICAL FOOTPRINT
# (what Activity Monitor calls Memory: resident + compressed + swapped
# anonymous) was ~16 GB -- killing that one process returned 16 GB of swap
# to a machine whose swap was 37.8 GB of 38.9 GB full and whose load average
# had reached 144. A concurrent agent's solve process, sampled at the same
# moment, read 5.2 GB resident against a 24.0 GB footprint (``vmmap
# --summary``: "Physical footprint: 24.0G, peak 32.9G").
#
# So: repeated solving in one process DOES grow the process, and under
# memory pressure the growth is invisible to RSS, because the cold leaked
# pages are exactly what the OS compresses and swaps first.
#
# An earlier revision of this file carried two contradictory claims about
# the growth (+3.6 GB per solve in the docstring, ~0.5 GB per solve in the
# budget comment). The three ``memory_probe`` series of section 0 settle
# it, and they show the two claims were not two readings of one number but
# two different EXECUTION PATHS:
#
#   * the jitted forward solve grows +3.351 GB of footprint and +2.495 GB
#     of resident per solve, which is the +3.6 GB claim, roughly right and
#     attached to the wrong generality;
#   * ``solve_spiral`` called directly does not grow at all (-0.052 GB per
#     solve over five, fit residual 0.124 GB), so no path in this study
#     reproduces a steady ~0.5 GB per solve;
#   * one ``value_and_grad`` pass is worse than either: +3.904 GB.
#
# Which path a block takes therefore decides its budget, and the rule of
# this machine allowed exactly one jitted solve per process. The cause was
# diagnosed afterwards by validation/fdfd/memory_probe.py (scipy SuperLU
# losing its factor when deallocated off its creating thread, triggered by
# XLA's callback threads) and fixed in rfx.fdfd.linear_solve; these series
# are the pre-fix record.
#
# ``footprint_gb`` reads ``ri_phys_footprint`` from ``proc_pid_rusage``
# (libSystem, flavour ``RUSAGE_INFO_V0``); it was CHECKED against ``vmmap
# --summary`` on a live 24 GB process and agreed to 0.03 GB (23.973 vs
# 24.0). ``mem_note`` records it after every solve, per block and per
# evaluation, and raises ``MemBudget`` the moment it crosses the limit --
# a clean stop that keeps the measurement, never an OOM kill and never a
# machine-wide swap storm. The check runs AFTER a solve, never inside one.
#
# The consequence for the study is the SOLVE BUDGET above: with the leak
# measured at MEM_GROWTH_GB_PER_SOLVE, a process gets at most a handful of
# solves, every block is resumable, and the gradient loop resumes by
# REPLAYING its own logged evaluations (see ``run_loop``).

MEM_LIMIT_GB = 6.0
EXIT_MEM = 4

# Every block that RUNS is measured under this instrumentation. One block is
# deliberately NOT RUN, and says so in the JSON (``not_run``) with its reason
# rather than being quietly absent; ``--mode-cost`` runs it anyway.
MEM_NOT_MEASURED: dict[str, str] = {}
BLOCKS_NOT_RUN = {
    "mode_cost": "the four-execution-mode cost block (forward / LU-cache hit / reverse / "
                 "three forward-mode jvps) has to run all four WITHIN one process, so it "
                 "cannot be chunked, and each factorising pass adds about 3.3 GB "
                 "(memory.probe_jit) to 4.0 GB (memory.probe_grad) of physical footprint: "
                 "2 rounds x 3 factorising passes puts it at 20-25 GB on a 38 GB machine "
                 "shared with three other agents, and this machine already thrashed once "
                 "today at 37.8 GB of swap. It is NOT run, and the cost claims of this "
                 "study are made instead from the factorisation COUNTS and the cache-hit "
                 "cost measured in one process by the `factorisations` block (2 solves) "
                 "plus the per-solve wall times logged by the loop, the sweep and the FD "
                 "stencil. Run it explicitly with --mode-cost --mem-limit 26 on an idle "
                 "machine if the four wall-time ratios are wanted.",
}
_MEM: dict[str, Any] = {"limit_gb": MEM_LIMIT_GB, "blocks": {}, "probe": None}
_T_PROC = time.time()


class MemBudget(RuntimeError):
    """The process physical footprint crossed ``MEM_LIMIT_GB``."""


def proc_memory(pid: int | None = None) -> dict[str, float]:
    """``proc_pid_rusage(RUSAGE_INFO_V0)``: resident size and PHYSICAL
    FOOTPRINT of a process, in GiB. The footprint is the number Activity
    Monitor shows and the number that decides whether this machine starts
    swapping; ``ru_maxrss`` and ``ps -o rss=`` both miss pages the OS has
    compressed or swapped out. Verified against ``vmmap --summary``
    (23.973 vs 24.0 GB on a live process). Returns NaNs off darwin."""
    import ctypes
    import sys
    if sys.platform != "darwin":                 # pragma: no cover - this machine is darwin
        import resource
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024.0 / 2.0 ** 30
        return {"resident_gb": rss, "footprint_gb": rss, "pageins": float("nan")}
    try:
        libc = ctypes.CDLL("/usr/lib/libSystem.B.dylib", use_errno=True)
        buf = (ctypes.c_uint8 * 512)()
        if libc.proc_pid_rusage(ctypes.c_int(os.getpid() if pid is None else pid),
                                ctypes.c_int(0), ctypes.byref(buf)) != 0:
            raise OSError("proc_pid_rusage failed")
        v = (ctypes.c_uint64 * 8).from_buffer(buf, 16)   # after ri_uuid[16]
        # v0 layout: user_time, system_time, pkg_idle_wkups, interrupt_wkups,
        #            pageins, wired_size, resident_size, phys_footprint
        return {"resident_gb": v[6] / 2.0 ** 30, "footprint_gb": v[7] / 2.0 ** 30,
                "pageins": float(v[4])}
    except Exception:                            # pragma: no cover - diagnostics only
        return {"resident_gb": float("nan"), "footprint_gb": float("nan"),
                "pageins": float("nan")}


def footprint_gb() -> float:
    return proc_memory()["footprint_gb"]


def maxrss_gb() -> float:
    """``ru_maxrss`` (bytes on darwin, KiB on linux) -- kept only because it
    is the number the earlier revision of this study quoted, so the two can
    be compared in the same record."""
    import resource
    import sys
    v = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return v / 2.0 ** 30 if sys.platform == "darwin" else v * 1024.0 / 2.0 ** 30


def mem_check(tag: str) -> dict[str, Any]:
    """Enforce the limit at a point where everything measured so far is
    already in the JSON. Every per-solve ``mem_note`` inside a block only
    RECORDS (``check=False``): a solve that has already been paid for -- 27 s
    at W/1, 230 s at W/2 -- must never be thrown away by the guard that is
    meant to protect the next one."""
    return mem_note(tag, check=True)


def mem_note(tag: str, store: list[dict[str, Any]] | None = None,
             check: bool = True) -> dict[str, Any]:
    """Record the process memory (and enforce ``MEM_LIMIT_GB`` on the
    footprint)."""
    m = proc_memory()
    r: dict[str, Any] = {"tag": tag, "footprint_gb": m["footprint_gb"], "resident_gb": m["resident_gb"],
         "maxrss_gb": maxrss_gb(), "solves": int(_BUDGET["used"]), "t": time.time() - _T_PROC}
    if store is not None:
        store.append(r)
    limit = float(_MEM.get("limit_gb", MEM_LIMIT_GB))
    if check and r["footprint_gb"] > limit:
        raise MemBudget(f"physical footprint {r['footprint_gb']:.2f} GB > {limit:.1f} GB limit "
                        f"after {r['solves']} solves ({tag}); resident {r['resident_gb']:.2f} GB, "
                        f"ru_maxrss {r['maxrss_gb']:.2f} GB -- rerun with a smaller --max-solves")
    return r


# ----------------------------------------------------------------------------
# 1. model

def stack(eps: bool = True, base_dz: float = BASE_DZ, metal_cells: int = METAL_CELLS):
    """The D2 stack geometry; ``eps=False`` sets the dielectrics to vacuum
    (only for the magnetoquasistatic referee cross-check)."""
    from rfx.fdfd import spiral as sm
    return sm.SmallStack(t_si=T_SI, t_ox_low=T_OX_LOW, t_m1=T_M1, t_via=T_VIA, t_m2=T_M2,
                         t_ox_high=T_OX_HIGH, t_air=T_AIR, base_dz=base_dz,
                         metal_cells=metal_cells,
                         eps_si=EPS_SI if eps else 1.0, eps_ox=EPS_OX if eps else 1.0)


def build_level(div: float = 1.0, eps: bool = True):
    """Static FDFD model at ``base_dx = WIDTH / div``, nominal theta."""
    from rfx.fdfd import spiral as sm
    spec = sm.SpiralSpec(n_turns=N_TURNS, r_out=R_OUT, spacing=SPACING, width=WIDTH, lead=LEAD,
                         base_dx=WIDTH / div, margin=MARGIN, stack=stack(eps),
                         pad_cells=PAD_CELLS, pad_ratio=PAD_RATIO)
    return sm.build_spiral(spec)


def model_record(model) -> dict[str, Any]:
    return {"shape": list(model.shape), "n_unknowns": int(model.n_unknowns),
            "base_dx": float(model.spec.dx),
            "wall_x": [float(model.x_nom[0]), float(model.x_nom[-1])],
            "wall_y": [float(model.y_nom[0]), float(model.y_nom[-1])],
            "lid_z": float(model.z[-1]), "build_seconds": float(model.build_seconds)}


def scale(theta0: Sequence[float] = THETA0) -> np.ndarray:
    """The optimiser works on ``u = theta / theta0`` (all three ~1), so the
    logged gradients and the FD steps are dimensionless."""
    return np.asarray(theta0, dtype=np.float64)


def bounds_scaled(bounds: Sequence[tuple[float, float]] = BOUNDS,
                  theta0: Sequence[float] = THETA0) -> list[tuple[float, float]]:
    s = scale(theta0)
    return [(lo / s[k], hi / s[k]) for k, (lo, hi) in enumerate(bounds)]


def feasibility_margin(model, theta) -> float:
    """``a_in - width/2`` (m), the binding topological constraint of a
    2-turn spiral (``spiral.check_feasible`` raises when it is <= 0)."""
    r_out, spacing, width = (float(v) for v in theta)
    return r_out - 3.0 * width - 2.0 * spacing


# ----------------------------------------------------------------------------
# 2. the traced objective

def metrics_fn(model, freq: float = FREQ, sigma_metal: float = SIGMA_CU,
               sigma_si: float = SIGMA_SI, theta0: Sequence[float] = THETA0) -> Callable:
    """``u -> (L_diff, Q_diff)``, traced, on the SCALED parameters."""
    import jax.numpy as jnp

    from rfx.fdfd import spiral as sm
    s = jnp.asarray(scale(theta0))

    def raw(u):
        r = sm.solve_spiral(model, freq, theta=jnp.asarray(u) * s,
                            sigma_metal=sigma_metal, sigma_si=sigma_si)
        return jnp.real(r.L_diff), jnp.real(r.Q_diff)

    return raw


def loss_of(l_value, q_value, l_target: float, q_ref: float, lam: float):
    return -q_value / q_ref + lam * ((l_value - l_target) / l_target) ** 2


def objective_fn(model, l_target: float, q_ref: float, lam: float,
                 **kw) -> tuple[Callable, Callable]:
    """``(value_and_grad, forward)``: both jitted, both on scaled ``u``.
    ``value_and_grad(u) -> ((loss, (L, Q)), grad)`` costs one three-fixture
    forward solve plus one adjoint pass that reuses its factorisations;
    ``forward(u) -> (loss, L, Q)`` costs the forward solve alone."""
    import jax
    raw = metrics_fn(model, **kw)

    def loss(u):
        l_value, q_value = raw(u)
        return loss_of(l_value, q_value, l_target, q_ref, lam), (l_value, q_value)

    def fwd(u):
        l_value, q_value = raw(u)
        return (loss_of(l_value, q_value, l_target, q_ref, lam), l_value, q_value)

    return jax.jit(jax.value_and_grad(loss, has_aux=True)), jax.jit(fwd)


def nominal_metrics(model) -> dict[str, Any]:
    """L and Q at the nominal theta (one forward solve) plus the Leontovich
    validity ratio -- the numbers L_target and Q_ref are built from."""
    import jax

    from rfx.fdfd import spiral as sm
    t0 = time.time()
    res = sm.solve_spiral(model, FREQ, sigma_metal=SIGMA_CU, sigma_si=SIGMA_SI)
    jax.block_until_ready(res.L_diff)
    sec = time.time() - t0
    return {"theta": list(THETA0), "L_diff": float(res.L_diff), "Q_diff": float(res.Q_diff),
            "L_se": float(res.L_se), "Q_se": float(res.Q_se), "L_raw": float(res.L_raw),
            "leontovich_validity": float(sm.leontovich_validity(model, FREQ, SIGMA_CU)),
            "skin_depth": float(sm.skin_depth(FREQ, SIGMA_CU)), "t_metal_min": float(T_M2),
            "forward_seconds": sec, "grid": model_record(model),
            "feasibility_margin_nominal": feasibility_margin(model, THETA0)}


def memory_probe(model, n_solves: int = 6, clear_cache: bool = True,
                 verbose: bool = True, mode: str = "forward") -> dict[str, Any]:
    """THE memory measurement this study's environment claim rests on.

    ``n_solves`` three-fixture forward solves of the SAME (nominal) theta in
    ONE process, with the LU factor cache emptied and ``gc.collect()`` forced
    between them, logging the PHYSICAL FOOTPRINT (and, for comparison, the
    resident size and ``ru_maxrss``) after each. Same theta + empty cache +
    forced collection means anything that grows is neither the working set,
    nor the cache, nor Python garbage.

    Why the footprint and not RSS: see section 0b. RSS is the number the two
    contradictory claims in the earlier revision of this file were built on,
    and under memory pressure it FALLS while the process keeps growing,
    because the cold pages get compressed and swapped.

    The growth is fitted over solves 2..n (solve 1 also pays for the model
    build and the jit compilation) and reported as
    ``growth_gb_per_solve``; the fit residual and the first/last values are
    recorded so the linearity can be judged rather than assumed.

    THREE modes, because they do not agree and the disagreement is the
    finding (see section 0b):

    * ``mode="forward"`` -- ``spiral.solve_spiral`` called directly, no
      ``jax.jit`` around it;
    * ``mode="jit"``  -- the SAME solve under ``jax.jit``, i.e. the path the
      loop, the sweep and the FD stencil all take;
    * ``mode="grad"`` -- ``jax.value_and_grad`` of it (one forward solve plus
      the adjoint pass), the shape of work the design loop does. That is the
      number that decides the loop's chunk size.
    """
    import gc

    import jax
    import jax.numpy as jnp

    from rfx.fdfd import spiral as sm
    from rfx.fdfd.linear_solve import clear_factor_cache
    vg = (jax.jit(jax.value_and_grad(lambda u_: metrics_fn(model)(u_)[0]))
          if mode == "grad" else None)
    fwd_jit = jax.jit(metrics_fn(model)) if mode == "jit" else None
    u_one = jnp.ones(3, dtype=jnp.float64)
    trace: list[dict[str, Any]] = []
    stopped_at_limit = False
    t_start = time.time()
    mem_note(f"probe-{mode}:start", trace, check=False)
    values: list[dict[str, Any]] = []
    for i in range(int(n_solves)):
        t0 = time.time()
        if vg is not None:                       # one forward + one adjoint
            l_value, grad = vg(u_one)
            jax.block_until_ready(grad)
            values.append({"i": i, "L": float(l_value), "Q": float("nan"),
                           "grad_norm": float(np.linalg.norm(np.asarray(grad))),
                           "seconds": time.time() - t0})
            del l_value, grad
        elif fwd_jit is not None:                # the jitted path the study runs on
            l_value, q_value = fwd_jit(u_one)
            jax.block_until_ready(l_value)
            values.append({"i": i, "L": float(l_value), "Q": float(q_value),
                           "seconds": time.time() - t0})
            del l_value, q_value
        else:
            res = sm.solve_spiral(model, FREQ, theta=jnp.asarray(THETA0),
                                  sigma_metal=SIGMA_CU, sigma_si=SIGMA_SI)
            jax.block_until_ready(res.L_diff)
            values.append({"i": i, "L": float(res.L_diff), "Q": float(res.Q_diff),
                           "seconds": time.time() - t0})
            del res
        if clear_cache:
            clear_factor_cache()
        gc.collect()
        # the probe MEASURES the limit, so it must not be stopped by it: it
        # records without checking and breaks itself when it crosses, which
        # keeps the trace that shows the crossing
        r = mem_note(f"probe-{mode}:solve{i + 1}", trace, check=False)
        if verbose:
            print(f"    solve {i + 1}: {values[-1]['seconds']:5.1f} s  footprint "
                  f"{r['footprint_gb']:5.2f} GB  resident {r['resident_gb']:4.2f} GB  "
                  f"ru_maxrss {r['maxrss_gb']:4.2f} GB", flush=True)
        if r["footprint_gb"] > float(_MEM["limit_gb"]):
            stopped_at_limit = True
            if verbose:
                print(f"    (stopping the probe: the limit {_MEM['limit_gb']:.1f} GB is "
                      f"crossed after {i + 1} solves)", flush=True)
            break
    after = [t for t in trace if ":solve" in t["tag"]]
    fps = [t["footprint_gb"] for t in after]
    rss = [t["resident_gb"] for t in after]
    mrs = [t["maxrss_gb"] for t in after]

    def fit(v: list[float]) -> tuple[float, float]:
        """Slope (GB per solve) over solves 2..n, and the max |residual|.
        With only two points the slope IS the increment between them (the
        reverse-mode probe crosses the limit after two passes and stops)."""
        if len(v) == 2:
            return float(v[1] - v[0]), 0.0
        if len(v) < 3:
            return float("nan"), float("nan")
        x = np.arange(len(v) - 1, dtype=float)
        a, b = np.polyfit(x, np.asarray(v[1:]), 1)
        return float(a), float(np.max(np.abs(np.asarray(v[1:]) - (a * x + b))))

    slope, resid = fit(fps)
    spread = max(v["L"] for v in values) / min(v["L"] for v in values) - 1.0 if values else 0.0
    return {"n_solves": len(values), "n_solves_requested": int(n_solves),
            "mode": mode, "stopped_at_limit": stopped_at_limit,
            "clear_cache": bool(clear_cache), "theta": list(THETA0),
            "grid": model_record(model), "trace": trace, "solves": values,
            "footprint_gb": fps, "resident_gb": rss, "maxrss_gb": mrs,
            "footprint_gb_max": max(fps) if fps else float("nan"),
            "footprint_gb_first_last": [fps[0], fps[-1]] if fps else None,
            "resident_gb_first_last": [rss[0], rss[-1]] if rss else None,
            "maxrss_gb_first_last": [mrs[0], mrs[-1]] if mrs else None,
            "growth_gb_per_solve": slope, "growth_fit_residual_gb": resid,
            "growth_gb_per_solve_resident": fit(rss)[0],
            "growth_gb_per_solve_maxrss": fit(mrs)[0],
            "solves_per_process_at_limit": (
                int(max(1.0, (float(_MEM["limit_gb"]) - fps[0]) / slope + 1.0))
                if fps and slope and slope > 1e-3 else None),
            # the same count against the DEFAULT 6 GB rule of this track, so a
            # probe that had to be run under a raised limit still reports the
            # number the rule implies
            "solves_per_process_at_default_limit": (
                int(max(1.0, (MEM_LIMIT_GB - fps[0]) / slope + 1.0))
                if fps and slope and slope > 1e-3 else None),
            "L_spread": float(spread),
            "limit_gb": float(_MEM["limit_gb"]), "seconds": time.time() - t_start,
            "note": "same theta, LU cache emptied and gc.collect() forced between solves; the "
                    "growth is fitted over solves 2..n (solve 1 also pays for the model build "
                    "and the jit compilation). The footprint is ri_phys_footprint from "
                    "proc_pid_rusage, checked against vmmap --summary to 0.03 GB"}


def choose_lambda(model, l_target: float, q_ref: float, verbose: bool = True,
                  **kw) -> dict[str, Any]:
    """Measure ``g_Q = d(Q/Q_ref)/du`` and ``g_L = d(L/L_t)/du`` at the
    nominal theta by two reverse-mode passes over ONE forward solve, and
    set ``lam = |g_Q| / (2 e_target |g_L|)`` (the bound on the relative L
    error at an interior stationary point; see the module doc)."""
    import jax
    import jax.numpy as jnp
    raw = metrics_fn(model, **kw)
    u0 = jnp.ones(3, dtype=jnp.float64)
    t0 = time.time()
    (l_value, q_value), pullback = jax.vjp(raw, u0)
    g_l = np.asarray(pullback((jnp.asarray(1.0) / l_target, jnp.asarray(0.0)))[0], dtype=np.float64)
    g_q = np.asarray(pullback((jnp.asarray(0.0), jnp.asarray(1.0) / q_ref))[0], dtype=np.float64)
    sec = time.time() - t0
    n_l, n_q = float(np.linalg.norm(g_l)), float(np.linalg.norm(g_q))
    lam = n_q / (2.0 * E_TARGET * n_l)
    rec = {"g_L_over_Lt": g_l.tolist(), "g_Q_over_Qref": g_q.tolist(),
           "norm_g_L": n_l, "norm_g_Q": n_q, "e_target": E_TARGET, "lam": lam,
           "seconds": sec, "L_at_nominal": float(l_value), "Q_at_nominal": float(q_value),
           "rule": "lam = |g_Q| / (2 e_target |g_L|): at an interior stationary point "
                   "-g_Q + 2 lam e g_L = 0 so |e| <= |g_Q| / (2 lam |g_L|) = e_target"}
    if verbose:
        print(f"  |g_Q| = {n_q:.4f}, |g_L| = {n_l:.4f} (scaled units)  ->  lam = {lam:.3f} "
              f"for e_target = {100 * E_TARGET:.0f} %", flush=True)
    return rec


# ----------------------------------------------------------------------------
# 3. the gradient loop

def run_loop(model, l_target: float, q_ref: float, lam: float, maxiter: int = MAXITER,
             gtol_factor: float = GTOL_FACTOR, dump: Callable[[], None] | None = None,
             record: dict[str, Any] | None = None, verbose: bool = True,
             theta0: Sequence[float] = THETA0,
             bounds: Sequence[tuple[float, float]] = BOUNDS,
             clear_cache: bool = True, resume: dict[str, Any] | None = None,
             **kw) -> dict[str, Any]:
    """L-BFGS-B on the scaled parameters with ``jac=True`` from
    ``jax.value_and_grad``. Every OBJECTIVE EVALUATION is logged (theta, L,
    Q, loss, |grad|); the ones scipy accepted as iterates are marked
    afterwards by matching the callback's x against the log (the line-search
    trials are the rest). ``dump`` is called after every evaluation so a
    killed run keeps its log.

    ``clear_cache`` (default ON) empties the LU factor cache of
    ``rfx.fdfd.linear_solve`` after each evaluation, and it is a MEMORY
    requirement, not an option: the cache is a 4-entry LRU and a W/1
    three-fixture ``value_and_grad`` fills three of those entries, so
    leaving it loaded across evaluations carries factorisations of the
    PREVIOUS theta into the next one. Measured on this machine
    (N = 33352): with the cache retained the process peak reached 6.33 GB
    by the third evaluation and tripped the 6.0 GB limit of section 0b;
    with it cleared, and one evaluation per process, no evaluation of the
    16 exceeded 5.29 GB. The
    price is the one theta scipy re-requests (it re-evaluates x0 after the
    warm-up call): with the cache live that repeat cost 0.5 s, with it
    cleared it costs a full solve. Nothing NUMERIC changes -- the solves
    are deterministic and the logged L, Q, loss and gradient are
    bit-identical either way (checked against the retained-cache run's
    first three evaluations: 0.0e0 relative).

    ``resume`` makes the loop CHUNKABLE, which the memory measurement of
    section 0b forces: the process footprint grows per solve, so a
    16-evaluation loop cannot live in one process under the 6 GB limit.
    L-BFGS-B holds its own state and cannot be pickled, so the loop is
    resumed by REPLAY: every evaluation is logged with its ``u``, loss and
    gradient, and a resumed run re-runs ``minimize`` from scratch with
    ``fun`` served from that log until it is exhausted, then continues with
    real solves until the solve budget stops it again. The replay is exact
    because the solves are deterministic -- and it is CHECKED, not assumed:
    the replayed requests must arrive in the logged order with bitwise
    equal ``u`` (``n_replay_mismatches`` in the record, measured 0 over the
    chunks of this study's run), otherwise the run aborts rather than
    silently optimising a different trajectory.
    """
    import gc

    import jax
    import jax.numpy as jnp
    from scipy.optimize import minimize

    from rfx.fdfd import spiral as sm
    from rfx.fdfd.linear_solve import clear_factor_cache

    vg, _ = objective_fn(model, l_target, q_ref, lam, theta0=theta0, **kw)
    s = scale(theta0)
    bnds = bounds_scaled(bounds, theta0)
    prior: list[dict[str, Any]] = [dict(e) for e in (resume or {}).get("evals", [])]
    evals: list[dict[str, Any]] = list(prior)
    queue: list[dict[str, Any]] = list(prior)    # replayed in order, then exhausted
    state = {"replayed": 0, "mismatches": 0, "new": 0}
    rec: dict[str, Any] = record if record is not None else {}
    rec.update({"clear_cache": bool(clear_cache),
                "lam": lam, "L_target": l_target, "Q_ref": q_ref, "maxiter": maxiter,
                "theta_nominal": list(theta0), "bounds": [list(b) for b in bounds],
                "bounds_scaled": bnds, "evals": evals, "complete": False,
                "n_evals_replayed": 0})
    t_start = time.time()
    if verbose and prior:
        print(f"    resuming by replay: {len(prior)} logged evaluations", flush=True)

    class _BudgetStop(Exception):
        """The solve budget ran out mid-loop; the log is kept and replayed."""

    def fun(u: np.ndarray):
        if queue:                                # REPLAY: no solve, bitwise check
            e = queue.pop(0)
            if [float(v) for v in u] != [float(v) for v in e["u"]]:
                state["mismatches"] += 1
                raise RuntimeError(
                    f"replay diverged at evaluation {e['i']}: L-BFGS-B asked for "
                    f"{list(np.asarray(u))} against the logged {e['u']}")
            state["replayed"] += 1
            rec["n_evals_replayed"] = state["replayed"]
            return float(e["loss"]), np.asarray(e["grad"], dtype=np.float64)
        if exhausted():
            raise _BudgetStop()
        t0 = time.time()
        theta = np.asarray(u, dtype=np.float64) * s
        sm.check_feasible(model, theta)          # the box is feasible by construction; assert it
        (loss, (l_value, q_value)), g = vg(jnp.asarray(u, dtype=jnp.float64))
        jax.block_until_ready(g)
        g = np.asarray(g, dtype=np.float64)
        evals.append({
            "i": len(evals), "u": [float(v) for v in u], "theta": theta.tolist(),
            "L": float(l_value), "Q": float(q_value), "loss": float(loss),
            "grad": g.tolist(), "grad_norm": float(np.linalg.norm(g)),
            "grad_inf": float(np.max(np.abs(g))),
            "L_rel_error": float(l_value) / l_target - 1.0,
            "feasibility_margin": feasibility_margin(model, theta),
            "seconds": time.time() - t0, "accepted": False,
        })
        # the loop is the one block that cannot be chunked (L-BFGS-B holds the
        # state), so it is also the one whose memory has to be watched from the
        # inside: rss_note raises MemBudget the moment the peak crosses the
        # limit, after the solve, so the trace that tripped it is kept
        if clear_cache:
            clear_factor_cache()                 # see the docstring: memory, not speed
            gc.collect()                         # and the reverse-mode residuals with it
        evals[-1].update(mem=mem_note(f"loop:eval{len(evals) - 1}", check=False))
        if verbose:
            e = evals[-1]
            print(f"    eval {e['i']:3d}  theta=({theta[0] * 1e6:6.3f}, {theta[1] * 1e6:6.3f}, "
                  f"{theta[2] * 1e6:6.3f}) um  L={float(l_value) * 1e12:8.3f} pH "
                  f"({100 * e['L_rel_error']:+6.2f} %)  Q={float(q_value):7.4f}  "
                  f"loss={float(loss):+.6f}  |g|={e['grad_norm']:.4g}  {e['seconds']:.1f} s",
                  flush=True)
        state["new"] += 1
        spend()
        if dump is not None:
            rec["seconds"] = rec.get("seconds_prior", 0.0) + time.time() - t_start
            dump()
        mem_check(f"loop:after-eval{len(evals) - 1}")   # the log is on disk; now guard
        return float(loss), g

    x0 = np.ones(3, dtype=np.float64)
    rec["seconds_prior"] = float((resume or {}).get("seconds", 0.0))
    f0, g0 = fun(x0)
    gtol = gtol_factor * float(np.max(np.abs(g0)))
    rec["gtol"] = gtol
    rec["grad_inf_at_nominal"] = float(np.max(np.abs(g0)))
    accepted_x: list[np.ndarray] = [x0.copy()]

    def cb(xk, *_a, **_k) -> None:
        accepted_x.append(np.asarray(xk, dtype=np.float64).copy())

    try:
        res = minimize(fun, x0, jac=True, method="L-BFGS-B", bounds=bnds, callback=cb,
                       options={"maxiter": maxiter, "gtol": gtol, "ftol": 1e-12, "maxcor": 10})
    except _BudgetStop:
        rec.update({"complete": False, "n_evals_logged": len(evals),
                    "n_new_solves_this_chunk": state["new"],
                    "n_replay_mismatches": state["mismatches"],
                    "seconds": rec.get("seconds_prior", 0.0) + time.time() - t_start})
        if verbose:
            print(f"    solve budget reached after {len(evals)} logged evaluations "
                  f"({state['new']} new this chunk); rerun to continue by replay", flush=True)
        return rec
    for xa in accepted_x:
        hit = [e for e in evals if np.allclose(np.asarray(e["u"]), xa, rtol=0, atol=1e-14)]
        if hit:
            hit[0]["accepted"] = True
    final = [e for e in evals if e["accepted"]][-1]
    n_fwd = len(evals)
    # scipy evaluates x0 again after the warm-up call that sets gtol. With the
    # LU cache live that repeat costs ~0.5 s (the cache is keyed by the matrix
    # entries and returns the same three factorisations); this loop clears the
    # cache between evaluations for memory, so the repeat costs a full solve.
    uniq = {tuple(np.round(np.asarray(e["u"]), 15)) for e in evals}
    rec.update({
        "complete": True, "n_evals_replayed": state["replayed"],
        "n_new_solves_this_chunk": state["new"], "n_replay_mismatches": state["mismatches"],
        "n_objective_evaluations": n_fwd,
        "n_forward_solves": n_fwd, "n_adjoint_passes": n_fwd,
        "n_distinct_thetas": len(uniq), "n_cached_repeats": n_fwd - len(uniq),
        "n_accepted_iterates": int(sum(e["accepted"] for e in evals)),
        "n_scipy_iterations": int(res.nit), "n_scipy_funcalls": int(res.nfev),
        "status": int(res.status), "message": str(res.message), "success": bool(res.success),
        "final": final,
        "final_u": final["u"], "final_theta": final["theta"],
        "final_L": final["L"], "final_Q": final["Q"], "final_loss": final["loss"],
        "final_L_rel_error": final["L_rel_error"], "final_grad_inf": final["grad_inf"],
        "loss_at_nominal": float(f0), "Q_at_nominal": q_ref,
        "seconds": rec.get("seconds_prior", 0.0) + time.time() - t_start,
    })
    if verbose:
        print(f"  L-BFGS-B: {res.nit} iterations, {n_fwd} objective evaluations "
              f"({res.message})", flush=True)
    return rec


# ----------------------------------------------------------------------------
# 4. the sweep baseline

def run_sweep(model, l_target: float, q_ref: float, lam: float, n: int = SWEEP_N,
              dump: Callable[[], None] | None = None, record: dict[str, Any] | None = None,
              verbose: bool = True, theta0: Sequence[float] = THETA0,
              bounds: Sequence[tuple[float, float]] = BOUNDS,
              resume: Sequence[dict[str, Any]] | None = None, **kw) -> dict[str, Any]:
    """``n^3`` forward solves on the tensor grid of the same box (no
    gradients). The baseline question is not "which point has the best
    loss" but "which points are even ON the constraint surface": the best
    Q inside the ``L_BAND`` band is what gate O3 compares against."""
    import gc

    import jax
    import jax.numpy as jnp

    from rfx.fdfd import spiral as sm
    from rfx.fdfd.linear_solve import clear_factor_cache
    _, fwd = objective_fn(model, l_target, q_ref, lam, theta0=theta0, **kw)
    s = scale(theta0)
    axes = [np.linspace(lo, hi, n) for lo, hi in bounds]
    pts: list[dict[str, Any]] = [dict(p) for p in (resume or [])]
    done = {tuple(round(float(v), 15) for v in p["theta"]) for p in pts}
    rec: dict[str, Any] = record if record is not None else {}
    rec.update({"n": n, "axes": [a.tolist() for a in axes], "band": L_BAND, "points": pts,
                "complete": False})
    t_start = time.time()
    if verbose and pts:
        print(f"    resuming with {len(pts)} of {n ** 3} points already in the JSON", flush=True)
    for a in axes[0]:
        for b in axes[1]:
            for c in axes[2]:
                theta = np.array([a, b, c], dtype=np.float64)
                if tuple(round(float(v), 15) for v in theta) in done:
                    continue
                if exhausted():
                    rec["seconds"] = rec.get("seconds", 0.0) + time.time() - t_start
                    return rec
                t0 = time.time()
                feasible = True
                try:
                    sm.check_feasible(model, theta)
                except ValueError:
                    feasible = False
                if feasible:
                    loss, l_value, q_value = fwd(jnp.asarray(theta / s, dtype=jnp.float64))
                    jax.block_until_ready(loss)
                    entry = {"theta": theta.tolist(), "L": float(l_value), "Q": float(q_value),
                             "loss": float(loss), "L_rel_error": float(l_value) / l_target - 1.0}
                else:
                    entry = {"theta": theta.tolist(), "L": None, "Q": None, "loss": None,
                             "L_rel_error": None}
                entry.update({"feasible": feasible, "seconds": time.time() - t0,
                              "feasibility_margin": feasibility_margin(model, theta)})
                pts.append(entry)
                if feasible:
                    spend()
                entry["mem"] = mem_note(f"sweep:{len(pts) - 1}", check=False)
                # each fixture's LU at N = 33352 is ~1 GB; nothing is reused
                # between sweep points, so release them (memory, not speed:
                # the cache would otherwise hold four of them). The
                # gc.collect() is NOT decoration: measured on this machine,
                # clearing the cache alone leaves ~2 GB per point behind
                # (a chunk crossed the 6 GB limit after 2 points, at 8.39 GB),
                # while clearing + collecting costs +0.25 GB per solve --
                # the rate memory_probe measures.
                clear_factor_cache()
                gc.collect()
                if verbose:
                    tag = (f"L={entry['L'] * 1e12:8.3f} pH ({100 * entry['L_rel_error']:+7.2f} %) "
                           f"Q={entry['Q']:7.4f} loss={entry['loss']:+.6f}"
                           if feasible else "INFEASIBLE")
                    print(f"    ({a * 1e6:5.2f}, {b * 1e6:5.2f}, {c * 1e6:5.2f}) um  {tag}  "
                          f"{entry['seconds']:.1f} s", flush=True)
                if dump is not None:
                    rec["seconds"] = time.time() - t_start
                    dump()
                mem_check(f"sweep:after-{len(pts) - 1}")
    rec["complete"] = True
    ok = [p for p in pts if p["feasible"]]
    rec["n_forward_solves"] = len(ok)
    rec["n_infeasible"] = len(pts) - len(ok)
    for band in (L_BAND, 0.05, 0.10, 0.25):
        inside = [p for p in ok if abs(p["L_rel_error"]) <= band]
        key = f"best_in_band_{band:g}"
        rec[key] = (max(inside, key=lambda p: p["Q"]) if inside else None)
        rec[f"n_in_band_{band:g}"] = len(inside)
    rec["best_loss"] = min(ok, key=lambda p: p["loss"]) if ok else None
    rec["best_Q_anywhere"] = max(ok, key=lambda p: p["Q"]) if ok else None
    rec["seconds"] = rec.get("seconds", 0.0) + time.time() - t_start
    return rec


# ----------------------------------------------------------------------------
# 5. the FD4 check at the final theta (gate O4)

def fd_check(model, u_final: Sequence[float], l_target: float, q_ref: float, lam: float,
             step: float = FD_STEP_REL, verbose: bool = True,
             theta0: Sequence[float] = THETA0, record: dict[str, Any] | None = None,
             **kw) -> dict[str, Any]:
    """4th-order central differences of the OBJECTIVE (and, from the same
    12 forward solves, of L and Q separately) against ``jax.grad`` at the
    final theta. The step is 1 % of the nominal parameter on the scaled
    variables -- the step D2 measured ``jax.grad`` against to 6e-7, and far
    above the ~1e-9 relative LU noise floor."""
    import gc

    import jax
    import jax.numpy as jnp

    from rfx.fdfd.linear_solve import clear_factor_cache
    vg, fwd = objective_fn(model, l_target, q_ref, lam, theta0=theta0, **kw)
    u0 = np.asarray(u_final, dtype=np.float64)
    t_start = time.time()
    rec: dict[str, Any] = record if record is not None else {}
    rec.setdefault("stencil", {})
    rec.update({"u": u0.tolist(), "theta": (u0 * scale(theta0)).tolist(), "step_rel": step,
                "fd_order": 4, "complete": False})
    stencil: dict[str, Any] = rec["stencil"]

    if "grad" not in rec:                        # the centre: one forward + one adjoint
        if exhausted():
            return rec
        (loss0, (l0, q0)), g = vg(jnp.asarray(u0))
        jax.block_until_ready(g)
        clear_factor_cache()
        gc.collect()
        spend()
        rec.update({"grad": np.asarray(g, dtype=np.float64).tolist(), "loss": float(loss0),
                    "L": float(l0), "Q": float(q0)})
    g = np.asarray(rec["grad"], dtype=np.float64)

    for k in range(3):
        for m in (2.0, 1.0, -1.0, -2.0):
            key = f"{k}:{m:+.0f}"
            if key in stencil:
                continue
            if exhausted():
                rec["seconds"] = rec.get("seconds", 0.0) + time.time() - t_start
                return rec
            u = u0.copy()
            u[k] += m * step
            out = fwd(jnp.asarray(u))
            jax.block_until_ready(out[0])
            clear_factor_cache()                 # see run_sweep: memory
            gc.collect()
            spend()
            stencil[key] = [float(v) for v in out]
            rec.setdefault("mem", []).append(mem_note(f"fd:{key}", check=False))
            if verbose:
                print(f"    stencil {key}: loss={stencil[key][0]:+.9f}", flush=True)

    fd: dict[str, list[float]] = {"loss": [], "L": [], "Q": []}
    for k in range(3):
        cols = [stencil[f"{k}:{m:+.0f}"] for m in (2.0, 1.0, -1.0, -2.0)]
        for j, name in enumerate(("loss", "L", "Q")):
            p2, p1, m1, m2 = (c[j] for c in cols)
            fd[name].append((-p2 + 8.0 * p1 - 8.0 * m1 + m2) / (12.0 * step))
        if verbose:
            print(f"    d/d{PARAMS[k]}: grad={g[k]:+.8e}  fd4={fd['loss'][k]:+.8e}  "
                  f"rel={abs(fd['loss'][k] - g[k]) / max(abs(g[k]), 1e-300):.3e}", flush=True)
    rel = [abs(fd["loss"][k] - g[k]) / max(abs(g[k]), 1e-300) for k in range(3)]
    rec.update({"fd4": fd["loss"], "rel": rel, "worst_rel": float(max(rel)),
                "fd4_L": fd["L"], "fd4_Q": fd["Q"], "complete": True,
                "seconds": rec.get("seconds", 0.0) + time.time() - t_start})
    return rec


# ----------------------------------------------------------------------------
# 5b. what a gradient actually costs here (the honest cost claim)

def mode_cost(model, l_target: float, q_ref: float, lam: float, u=None,
              verbose: bool = True, rounds: int = 2) -> dict[str, Any]:
    """MEASURE the costs the study's message rests on, at the design point,
    with the LU cache cleared before each block and the jit compilation
    excluded (every function is compiled AHEAD OF TIME with
    ``jax.jit(...).lower(...).compile()``, which compiles without executing
    a solve -- the warm-up call would otherwise cost a full factorisation):

    * ``forward``          -- the three fixtures, value only;
    * ``forward_cached``   -- the SAME point again, served by the LU cache;
    * ``value_and_grad``   -- one forward plus the reverse pass, which
      reuses the three factorisations through that cache;
    * ``forward_mode_3``   -- three ``jax.jvp`` calls, one per parameter,
      i.e. the whole Jacobian in forward mode (``jax.jacfwd`` cannot be
      used: it vmaps the ``pure_callback`` of ``linear_solve`` and jax
      refuses). The first call factorises, the other two hit the cache.

    With THREE parameters forward mode is not asymptotically worse, and
    this is the number that says so instead of a hand-wave. Only three of
    the calls actually factorise, so the block costs ~3 solves."""
    import jax
    import jax.numpy as jnp

    from rfx.fdfd.linear_solve import clear_factor_cache
    raw = metrics_fn(model)

    def loss_only(u_):
        l_value, q_value = raw(u_)
        return loss_of(l_value, q_value, l_target, q_ref, lam)

    u0 = jnp.asarray(np.ones(3) if u is None else np.asarray(u, dtype=np.float64))
    eye = [jnp.asarray(v, dtype=jnp.float64) for v in np.eye(3)]
    t0 = time.time()
    fwd = jax.jit(loss_only).lower(u0).compile()
    vg = jax.jit(jax.value_and_grad(loss_only)).lower(u0).compile()
    jvp = jax.jit(lambda u_, t_: jax.jvp(loss_only, (u_,), (t_,))).lower(u0, eye[0]).compile()
    compile_seconds = time.time() - t0

    # One throwaway EXECUTION before the timings: the first solve of a
    # process is slower than the steady state (it pages in the working set,
    # starts the BLAS threads and loads SuperLU), and -- far worse on this
    # shared machine -- the three blocks must be INTERLEAVED. Measured in a
    # single sequential pass, "forward 39.2 s, value_and_grad 26.1 s, three
    # jvps 16.4 s" -- monotonically decreasing, i.e. the other agents on the
    # box going idle, not an algorithmic ordering (one reverse pass cannot
    # be cheaper than the forward solve it differentiates). So the block is
    # run in ROUNDS and the per-round ratios, which see the same load, are
    # what the study quotes; the raw per-round seconds and ``os.getloadavg``
    # are recorded so the contention is visible.
    out: dict[str, Any] = {"u": [float(v) for v in u0], "n_unknowns": int(model.n_unknowns),
                           "compile_seconds": compile_seconds, "rounds": []}
    t0 = time.time()
    jax.block_until_ready(fwd(u0))
    out["first_call_seconds"] = time.time() - t0

    for r in range(int(rounds)):
        rec: dict[str, Any] = {"round": r, "loadavg_before": list(os.getloadavg())}
        clear_factor_cache()
        t0 = time.time()
        jax.block_until_ready(fwd(u0))
        rec["forward"] = time.time() - t0
        t0 = time.time()
        jax.block_until_ready(fwd(u0))
        rec["forward_cached"] = time.time() - t0       # same point, LU cache hit
        clear_factor_cache()
        t0 = time.time()
        jax.block_until_ready(vg(u0))
        rec["value_and_grad"] = time.time() - t0
        clear_factor_cache()
        t0 = time.time()
        for e in eye:
            jax.block_until_ready(jvp(u0, e))
        rec["forward_mode_3"] = time.time() - t0
        clear_factor_cache()
        rec["adjoint_overhead"] = rec["value_and_grad"] / rec["forward"] - 1.0
        rec["forward_mode_over_reverse"] = rec["forward_mode_3"] / rec["value_and_grad"]
        rec["cached_over_full"] = rec["forward_cached"] / rec["forward"]
        rec["loadavg_after"] = list(os.getloadavg())
        out["rounds"].append(rec)
        if verbose:
            print(f"    round {r}: forward {rec['forward']:.1f} s | cached repeat "
                  f"{rec['forward_cached']:.2f} s ({100 * rec['cached_over_full']:.1f} %) | "
                  f"value_and_grad {rec['value_and_grad']:.1f} s "
                  f"({100 * rec['adjoint_overhead']:+.0f} %) | 3 jvps "
                  f"{rec['forward_mode_3']:.1f} s ({rec['forward_mode_over_reverse']:.2f} x "
                  f"reverse) | load {rec['loadavg_before'][0]:.1f}", flush=True)

    for key in ("forward", "forward_cached", "value_and_grad", "forward_mode_3",
                "adjoint_overhead", "forward_mode_over_reverse", "cached_over_full"):
        out[key] = float(np.median([r[key] for r in out["rounds"]]))
    # the ratios are medians of the PER-ROUND ratios (each round sees one
    # load level), not ratios of the median seconds
    out["fd4_equivalent_forward_solves"] = 12
    out["fd4_estimated_seconds"] = 12.0 * out["forward"]
    if verbose:
        print(f"    median of {rounds} rounds: forward {out['forward']:.1f} s, adjoint "
              f"+{100 * out['adjoint_overhead']:.0f} %, forward mode "
              f"{out['forward_mode_over_reverse']:.2f} x reverse, cached repeat "
              f"{100 * out['cached_over_full']:.1f} % of a solve; FD4 would be "
              f"{out['fd4_estimated_seconds']:.0f} s", flush=True)
    return out


def factorisation_accounting(model, l_target: float, q_ref: float, lam: float,
                             u=None, verbose: bool = True) -> dict[str, Any]:
    """Count the LU FACTORISATIONS behind the wall times, by wrapping
    ``rfx.fdfd.linear_solve._factor`` (instrumentation only -- the wrapper
    is removed again and nothing in ``rfx`` is modified). This is what
    makes the cost claim defensible on a contended machine, because it
    counts work instead of measuring seconds:

    * a forward solve calls ``_factor`` three times, all three MISS: the
      three fixtures are factorised, and the summed factorisation time
      equals the wall time (measured 39.2 of 39.6 s), i.e. the solve IS the
      three factorisations;
    * ``value_and_grad`` calls it six times with three misses: the adjoint
      adds no factorisation at all, only three transposed back-substitutions
      on the cached factors. Its wall time came out BELOW the forward
      solve's (26.6 s against 39.6 s) although the summed factorisation
      time is the same 39.3 s -- the reverse-mode graph has enough
      independent work for XLA to run the three host callbacks CONCURRENTLY
      (SuperLU releases the GIL), while the forward-only graph serialises
      them. That is a scheduling artefact of this backend, not a property
      of adjoints, and it is the reason the study quotes the factorisation
      COUNT and the cache-hit cost rather than the wall-time ratio.
    """
    import jax
    import jax.numpy as jnp

    from rfx.fdfd import linear_solve as ls
    raw = metrics_fn(model)

    def loss_only(u_):
        l_value, q_value = raw(u_)
        return loss_of(l_value, q_value, l_target, q_ref, lam)

    u0 = jnp.asarray(np.ones(3) if u is None else np.asarray(u, dtype=np.float64))
    fwd = jax.jit(loss_only).lower(u0).compile()
    vg = jax.jit(jax.value_and_grad(loss_only)).lower(u0).compile()

    original = ls._factor
    counter = {"calls": 0, "misses": 0, "factor_seconds": 0.0}

    def counted(*a: Any, **k: Any):
        counter["calls"] += 1
        n0 = len(ls._FACTOR_CACHE)
        t0 = time.time()
        out = original(*a, **k)
        if len(ls._FACTOR_CACHE) > n0:
            counter["misses"] += 1
            counter["factor_seconds"] += time.time() - t0
        return out

    out: dict[str, Any] = {"u": [float(v) for v in u0]}
    try:
        ls._factor = counted                     # instrumentation, restored below
        # "forward_cached" deliberately does NOT clear the cache: it is the
        # same point solved again, so every _factor call hits and what is left
        # is the three back-substitutions. That is the cost a forward-mode
        # tangent pays on top of a solve, measured instead of asserted, and it
        # costs no factorisation and no extra memory.
        for name, fn, clear in (("forward", fwd, True), ("forward_cached", fwd, False),
                                ("value_and_grad", vg, True)):
            if clear:
                ls.clear_factor_cache()
            counter.update(calls=0, misses=0, factor_seconds=0.0)
            t0 = time.time()
            jax.block_until_ready(fn(u0))
            out[name] = {"wall_seconds": time.time() - t0, **dict(counter)}
            if verbose:
                c = out[name]
                print(f"    {name:15s} wall {c['wall_seconds']:5.1f} s  _factor calls "
                      f"{c['calls']}  misses {c['misses']}  factorisation "
                      f"{c['factor_seconds']:.1f} s", flush=True)
    finally:
        ls._factor = original
        ls.clear_factor_cache()
    out["extra_factorisations_for_the_gradient"] = (out["value_and_grad"]["misses"]
                                                    - out["forward"]["misses"])
    out["cached_over_full"] = (out["forward_cached"]["wall_seconds"]
                               / out["forward"]["wall_seconds"])
    out["fd4_equivalent_forward_solves"] = 12
    out["factor_over_wall_forward"] = (out["forward"]["factor_seconds"]
                                       / out["forward"]["wall_seconds"])
    out["concurrency_in_value_and_grad"] = (out["value_and_grad"]["factor_seconds"]
                                            / out["value_and_grad"]["wall_seconds"])
    return out


# ----------------------------------------------------------------------------
# 6. the finer-grid re-solve (gate O5, report only)

def refine(theta_final: Sequence[float], l_target: float, q_ref: float, lam: float,
           coarse: dict[str, Any], div: float = 2.0, verbose: bool = True,
           record: dict[str, Any] | None = None) -> dict[str, Any]:
    """Re-solve the NOMINAL and the FINAL theta at ``base_dx = WIDTH/div``
    (forward only -- no gradient, to keep the peak memory down) and report
    the shifts. The nominal point is re-solved too so the COMMON-MODE part
    of the shift can be separated from the part that moves the design."""
    import gc

    import jax
    import jax.numpy as jnp

    from rfx.fdfd.linear_solve import clear_factor_cache
    t_start = time.time()
    clear_factor_cache()                         # the W/2 factors are ~3x the W/1 ones
    model = build_level(div)
    _, fwd = objective_fn(model, l_target, q_ref, lam)
    s = scale(THETA0)
    out: dict[str, Any] = record if record is not None else {}
    out.setdefault("points", {})
    out.update({"div": div, "grid": model_record(model), "complete": False})
    if verbose:
        print(f"  W/{div:g}: {model.shape} N={model.n_unknowns}", flush=True)
    for name, theta in (("nominal", np.asarray(THETA0)), ("final", np.asarray(theta_final))):
        if name in out["points"]:
            continue
        if exhausted():
            out["seconds"] = out.get("seconds", 0.0) + time.time() - t_start
            return out
        t0 = time.time()
        loss, l_value, q_value = fwd(jnp.asarray(np.asarray(theta) / s))
        jax.block_until_ready(loss)
        clear_factor_cache()
        gc.collect()
        spend(3)                                 # a W/2 solve is ~3x the W/1 memory
        out["points"][name] = {"theta": np.asarray(theta).tolist(), "L": float(l_value),
                               "Q": float(q_value), "loss": float(loss),
                               "L_rel_error": float(l_value) / l_target - 1.0,
                               "seconds": time.time() - t0,
                               "mem": mem_note(f"refine:{name}", check=False)}
        if verbose:
            p = out["points"][name]
            print(f"    {name:8s} L={p['L'] * 1e12:8.3f} pH  Q={p['Q']:7.4f}  "
                  f"({p['seconds']:.0f} s)", flush=True)
    for name in ("nominal", "final"):
        c = coarse[name]
        f = out["points"][name]
        out["points"][name].update({
            "L_coarse": c["L"], "Q_coarse": c["Q"],
            "L_shift": f["L"] / c["L"] - 1.0, "Q_shift": f["Q"] / c["Q"] - 1.0})
    out["complete"] = True
    out["seconds"] = out.get("seconds", 0.0) + time.time() - t_start
    return out


# ----------------------------------------------------------------------------
# 7. the referee cross-check of the ABSOLUTE bias at THIS geometry

def referee_bias(div: float = 1.0, verbose: bool = True) -> dict[str, Any]:
    """D2's V1 comparison, repeated at THIS study's nominal geometry: the
    uniform-current volumetric-metal FDFD (vacuum dielectrics, the only
    configuration the magnetoquasistatic Greenhouse referee can be compared
    to) against the area-exact referee. One forward solve. The number this
    returns is the honest size of the absolute-L bias the design loop runs
    on -- measured here, not carried over from D2's r_out = 52 um fixture.
    Uses the referee helpers of ``spiral_convergence.py`` (loaded by path;
    the validation directory is not a package)."""
    import jax
    import jax.numpy as jnp

    from rfx.fdfd import spiral as sm
    spec = importlib.util.spec_from_file_location("spiral_convergence", CONV_PATH)
    assert spec is not None and spec.loader is not None
    conv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(conv)
    sg = conv.load_referee()
    t_start = time.time()
    ref = {}
    for name, kw in (("area_exact", {"convention": "area_exact"}),
                     ("greenhouse", {"convention": "greenhouse"})):
        ref[name] = float(conv.referee_value(sg, THETA0, lead=LEAD, shift=0.5 * WIDTH,
                                             ground=GROUND_H, **kw).total)
    model = build_level(div, eps=False)
    t0 = time.time()
    res = sm.solve_spiral(model, conv.FREQ, theta=jnp.asarray(THETA0),
                          sigma_volumetric=SIGMA_VOL)
    jax.block_until_ready(res.L_diff)
    l_fdfd = float(res.L_diff)
    out: dict[str, Any] = {"protocol": "D2 V1 at this study's geometry: uniform-current volumetric metal "
                       f"(sigma={SIGMA_VOL:g} S/m) at {conv.FREQ:g} Hz, vacuum dielectrics, "
                       "pad_cells=4, area-exact corner convention on the referee",
           "freq": float(conv.FREQ), "sigma_volumetric": SIGMA_VOL, "div": div,
           "referee": ref, "corner_systematic": ref["area_exact"] / ref["greenhouse"] - 1.0,
           "L_fdfd": l_fdfd, "gap_vs_area_exact": l_fdfd / ref["area_exact"] - 1.0,
           "grid": model_record(model), "solve_seconds": time.time() - t0,
           "seconds": time.time() - t_start,
           "d2_gap_at_W1": -0.2494}
    if verbose:
        print(f"  referee (area-exact) {ref['area_exact'] * 1e12:.3f} pH, FDFD uniform-current "
              f"{l_fdfd * 1e12:.3f} pH  ->  {100 * out['gap_vs_area_exact']:+.2f} % "
              f"(D2 measured {100 * out['d2_gap_at_W1']:+.2f} % at W/1 on its r_out = 52 um "
              f"fixture)", flush=True)
    return out


# ----------------------------------------------------------------------------
# 8. gates

def projected_gradient(u: Sequence[float], grad: Sequence[float],
                       bnds: Sequence[Sequence[float]]) -> np.ndarray:
    """``u - clip(u - g, lo, hi)``: the quantity L-BFGS-B's ``gtol`` tests.
    At an ACTIVE bound the outward component is projected away, so the
    full ``|grad|`` logged per evaluation stays finite while the optimiser
    is converged -- the distinction matters here because the optimum sits
    on the ``width`` bound."""
    u = np.asarray(u, dtype=np.float64)
    g = np.asarray(grad, dtype=np.float64)
    lo = np.asarray([b[0] for b in bnds], dtype=np.float64)
    hi = np.asarray([b[1] for b in bnds], dtype=np.float64)
    return u - np.clip(u - g, lo, hi)


def evaluate_gates(study: dict[str, Any]) -> dict[str, Any]:
    g: dict[str, Any] = {}
    loop = study.get("loop", {})
    evals = loop.get("evals", [])
    acc = [e for e in evals if e.get("accepted")]

    if acc:
        bnds = loop.get("bounds_scaled", bounds_scaled())
        pg = [float(np.max(np.abs(projected_gradient(e["u"], e["grad"], bnds)))) for e in acc]
        at_bound = [[bool(abs(e["u"][k] - bnds[k][0]) < 1e-12 or abs(e["u"][k] - bnds[k][1]) < 1e-12)
                     for k in range(3)] for e in acc]
        g["stopping"] = {
            "name": "the |grad| stop criterion (scipy tests the PROJECTED gradient)",
            "gtol": loop.get("gtol"), "gtol_factor": study["fixture"]["gtol_factor"],
            "projected_grad_inf_per_accepted_iterate": pg,
            "projected_grad_inf_final": pg[-1],
            "full_grad_inf_final": acc[-1]["grad_inf"],
            "active_bounds_final": dict(zip(PARAMS, at_bound[-1])),
            "message": loop.get("message"), "n_scipy_iterations": loop.get("n_scipy_iterations"),
            "passed": None}

    losses = [e["loss"] for e in acc]
    worst_rise = max([losses[i + 1] - losses[i] for i in range(len(losses) - 1)], default=0.0)
    g["O1"] = {"name": "loss decreases over the accepted iterates",
               "n_accepted": len(acc), "losses": losses,
               "worst_increase": worst_rise, "threshold": 0.0,
               "passed": bool(len(acc) >= 2 and worst_rise <= 0.0)}

    e_fin = abs(loop.get("final_L_rel_error", float("nan")))
    g["O2"] = {"name": "final |L - L_target| / L_target <= 2 %",
               "measured": e_fin, "threshold": L_BAND, "passed": bool(e_fin <= L_BAND)}

    sweep = study.get("sweep", {})
    best = sweep.get(f"best_in_band_{L_BAND:g}")
    q_fin = loop.get("final_Q")
    # the COST comparison, summed from the per-solve seconds the blocks
    # logged. A block's own "seconds" field is the chunk that wrote it (the
    # study runs as resumable chunks), so it is not the study's cost; the sum
    # over the logged solves is.
    cost = {"loop_evaluations": len(evals),
            "loop_solve_seconds": float(sum(e["seconds"] for e in evals)),
            "sweep_forward_solves": sweep.get("n_forward_solves"),
            "sweep_solve_seconds": float(sum(p["seconds"] for p in sweep.get("points", []))),
            "fd4_gradient_forward_solves": len((study.get("fd_check") or {}).get("stencil", {})),
            "fd4_gradient_seconds": (study.get("fd_check") or {}).get("seconds")}
    if best is None:
        wider = {b: sweep.get(f"best_in_band_{b:g}") for b in (0.05, 0.10, 0.25)}
        qualifying = {f"{b:g}": (w["Q"] if w else None) for b, w in wider.items()}
        g["O3"] = {"name": "final Q >= best sweep Q inside the 2 % L band",
                   "sweep_points_in_band": sweep.get(f"n_in_band_{L_BAND:g}", 0),
                   "statement": "NO sweep point qualifies: none of the "
                                f"{sweep.get('n_forward_solves')} forward solves of the "
                                f"{SWEEP_N}x{SWEEP_N}x{SWEEP_N} grid lands inside the 2 % L band",
                   "final_Q": q_fin, "best_sweep_Q_in_wider_bands": qualifying,
                   "best_sweep_Q_anywhere": (sweep.get("best_Q_anywhere") or {}).get("Q"),
                   "best_sweep_loss": (sweep.get("best_loss") or {}).get("loss"),
                   "final_loss": loop.get("final_loss"), **cost,
                   "passed": None}
    else:
        g["O3"] = {"name": "final Q >= best sweep Q inside the 2 % L band",
                   "sweep_points_in_band": sweep.get(f"n_in_band_{L_BAND:g}"),
                   "final_Q": q_fin, "best_sweep_Q": best["Q"], "best_sweep_theta": best["theta"],
                   "best_sweep_L_rel_error": best["L_rel_error"],
                   "margin": (q_fin - best["Q"]) / best["Q"] if q_fin is not None else None,
                   **cost,
                   "passed": bool(q_fin is not None and q_fin >= best["Q"])}

    fd = study.get("fd_check")
    if fd and fd.get("complete"):
        grad = np.asarray(fd["grad"], dtype=np.float64)
        fd4 = np.asarray(fd["fd4"], dtype=np.float64)
        err = np.abs(fd4 - grad)
        norm = float(np.max(np.abs(grad)))
        # PER-COMPONENT relative agreement is not a usable metric at a
        # CONSTRAINED optimum: the free-direction component of the gradient
        # is zero there by stationarity (measured dloss/dr_out = 8.61e-5,
        # 2.2e-4 of the gradient's sup norm 0.399), so dividing an absolute
        # FD-truncation error of ~3e-8 by it gives 3.5e-3 while the two
        # bound-active components agree to 3.4e-7 and 1.1e-7. The gate is
        # therefore stated on the error relative to the gradient NORM,
        # which is the quantity an optimiser actually consumes; both forms
        # are recorded. The absolute errors (3.0e-7 / 6.1e-9 / 4.3e-8) are
        # all at the same level, i.e. the FD4 truncation floor of this
        # 1 %-step stencil, not a gradient defect.
        g["O4"] = {"name": "jax.grad vs FD4 at the final theta (objective)",
                   "rel_per_component": fd["rel"], "worst_rel_per_component": fd["worst_rel"],
                   "abs_err": err.tolist(), "worst_abs_err": float(np.max(err)),
                   "grad_inf_norm": norm, "rel_to_grad_norm": float(np.max(err) / norm),
                   "step_rel": fd["step_rel"], "threshold": 1e-4,
                   "smallest_component_over_norm": float(np.min(np.abs(grad)) / norm),
                   "passed": bool(float(np.max(err) / norm) <= 1e-4)}

    ref = study.get("refine")
    if ref:
        p = ref["points"]
        # The two shifts split into a COMMON part (the grid bias, which moves
        # every point of the box the same way) and a DIFFERENTIAL part (the
        # part that would move the design). Reported as a fraction, because
        # that fraction is the whole claim of this block: 0.050 measured, i.e.
        # 95 % of the W/1 -> W/2 move is common mode.
        d_shift = p["final"]["L_shift"] - p["nominal"]["L_shift"]
        big = max(abs(p["final"]["L_shift"]), abs(p["nominal"]["L_shift"]))
        conv_path = HERE / "spiral_convergence.json"
        d2_implied = None
        if conv_path.exists():                   # D2's own W/1 -> W/2 move, for CONTEXT only
            lv = json.loads(conv_path.read_text()).get("levels", {})
            if "1" in lv and "2" in lv:
                d2_implied = lv["2"]["L_dut"] / lv["1"]["L_dut"] - 1.0
        g["O5"] = {"name": "W/2 re-solve (REPORT ONLY -- no pass/fail is meaningful)",
                   "L_shift_final": p["final"]["L_shift"], "Q_shift_final": p["final"]["Q_shift"],
                   "L_shift_nominal": p["nominal"]["L_shift"],
                   "Q_shift_nominal": p["nominal"]["Q_shift"],
                   "L_shift_differential": d_shift,
                   "common_mode_fraction": 1.0 - abs(d_shift) / big if big else None,
                   "d2_implied_shift_W1_to_W2": d2_implied,
                   "same_sign_as_d2": (None if d2_implied is None else
                                       bool(d2_implied > 0 and p["final"]["L_shift"] > 0
                                            and p["nominal"]["L_shift"] > 0)),
                   "L_rel_error_at_W2": p["final"]["L_rel_error"],
                   "L_target_W1": study["loop"]["L_target"],
                   "note": "the coarse model is known to be 15-25 % low on absolute L (D2 gate "
                           "V1 FAILED and the referee cross-check in referee_bias repeats it "
                           "here); a shift of that order is expected, which is why this gate "
                           "reports and does not judge. D2's own W/1 -> W/2 move is quoted for "
                           "context and NOT as a window: it was measured on a different "
                           "geometry (r_out = 52 um, vacuum dielectrics, uniform-current "
                           "metal), so only its SIGN is a prediction for this fixture",
                   "passed": None}

    def per_process_at_default(rec: dict[str, Any]) -> int | None:
        """Solves one process gets under the DEFAULT 6 GB rule, recomputed
        here from the probe's raw series so a probe that had to be run under
        a raised limit still reports the number the rule implies."""
        fps = rec.get("footprint_gb") or []
        slope = rec.get("growth_gb_per_solve")
        if not fps or not slope or slope <= 1e-3:
            return None
        return int(max(1.0, (MEM_LIMIT_GB - fps[0]) / slope + 1.0))

    # ---- M1: the memory rule this machine imposes (see section 0b) ---------
    mem = study.get("memory") or {}
    blocks = mem.get("by_block") or {}
    probe = mem.get("probe") or {}
    probe_jit = mem.get("probe_jit") or {}

    def logged_peak(record: Any) -> float:
        """The highest footprint any ``mem_note`` INSIDE a block recorded.

        The per-chunk numbers are taken at the END of a chunk, after the
        factor cache is dropped and ``gc.collect()`` has run, so they
        understate what the block reached: the W/2 re-solve ends a chunk at
        14.01 GB having passed through 18.21 GB, and one sweep process ends
        at 3.72 GB having reached 8.56 GB. A block's peak is therefore the
        max over its own logged notes as well."""
        best = float("nan")
        stack: list[Any] = [record]
        while stack:
            node = stack.pop()
            if isinstance(node, dict):
                v = node.get("footprint_gb")
                if isinstance(v, (int, float)) and not (best == best and v <= best):
                    best = float(v)
                stack.extend(node.values())
            elif isinstance(node, list):
                stack.extend(node)
        return best

    # the probe blocks keep their trace under memory.probe*, not under a
    # top-level key of the study
    sources = dict(study, memory_probe=probe, memory_probe_jit=probe_jit,
                   memory_probe_grad=mem.get("probe_grad") or {})
    peaks = {}
    for k, v in blocks.items():
        chunk_peak = float(v.get("footprint_gb_max", v.get("footprint_gb_after", float("nan"))))
        inner = logged_peak(sources.get(k))
        peaks[k] = max(chunk_peak, inner) if inner == inner else chunk_peak
    limits = {k: float(v.get("limit_gb", MEM_LIMIT_GB)) for k, v in blocks.items()}
    over = {k: [peaks[k], limits[k]] for k in peaks if peaks[k] > limits[k]}
    # the memory probes are the measurement OF the limit: the reverse-mode one
    # is supposed to cross it (it records the crossing and stops itself), so it
    # is listed separately from the blocks that crossed it by accident. The
    # gate judges only the latter.
    by_design = {k: v for k, v in over.items() if k.startswith("memory_probe")}
    unintended = {k: v for k, v in over.items() if k not in by_design}
    missing = sorted(k for k, v in study.items()
                     if isinstance(v, dict) and k not in peaks and k not in
                     ("fixture", "design", "memory", "gates", "seconds_by_block", "not_run"))
    g["M1"] = {"name": "physical footprint of every block within the limit it ran under, "
                       "and the per-solve growth measured rather than assumed",
               "metric": "ri_phys_footprint (proc_pid_rusage), NOT ru_maxrss -- see the "
                         "study's section 0b: under memory pressure RSS falls while the "
                         "process grows",
               "footprint_gb_by_block": peaks, "limit_gb_by_block": limits,
               "blocks_over_limit": unintended,
               "blocks_over_limit_by_design": by_design,
               "blocks_not_measured": missing,
               "why_not_measured": mem.get("not_measured") or {},
               "probe_n_solves": probe.get("n_solves"),
               "probe_footprint_gb_first_last": probe.get("footprint_gb_first_last"),
               "probe_resident_gb_first_last": probe.get("resident_gb_first_last"),
               "probe_growth_gb_per_solve": probe.get("growth_gb_per_solve"),
               "probe_growth_gb_per_solve_resident": probe.get("growth_gb_per_solve_resident"),
               "probe_growth_fit_residual_gb": probe.get("growth_fit_residual_gb"),
               "probe_solves_per_process_at_limit": probe.get("solves_per_process_at_limit"),
               # the same series on the JITTED path -- the one the loop, the
               # sweep and the FD stencil actually take, and the only one of
               # the three that grows (section 0b)
               "probe_jit_n_solves": probe_jit.get("n_solves"),
               "probe_jit_footprint_gb_first_last": probe_jit.get("footprint_gb_first_last"),
               "probe_jit_resident_gb_first_last": probe_jit.get("resident_gb_first_last"),
               "probe_jit_growth_gb_per_solve": probe_jit.get("growth_gb_per_solve"),
               "probe_jit_growth_gb_per_solve_resident":
                   probe_jit.get("growth_gb_per_solve_resident"),
               "probe_jit_growth_fit_residual_gb": probe_jit.get("growth_fit_residual_gb"),
               "probe_jit_solves_per_process_at_limit":
                   probe_jit.get("solves_per_process_at_limit"),
               "probe_jit_solves_per_process_at_default_limit":
                   per_process_at_default(probe_jit),
               "probe_jit_limit_gb": probe_jit.get("limit_gb"),
               "probe_grad_n_passes": (mem.get("probe_grad") or {}).get("n_solves"),
               "probe_grad_footprint_gb_first_last":
                   (mem.get("probe_grad") or {}).get("footprint_gb_first_last"),
               "probe_grad_growth_gb_per_pass":
                   (mem.get("probe_grad") or {}).get("growth_gb_per_solve"),
               "probe_grad_stopped_at_limit":
                   (mem.get("probe_grad") or {}).get("stopped_at_limit"),
               "probe_grad_passes_per_process_at_limit":
                   (mem.get("probe_grad") or {}).get("solves_per_process_at_limit"),
               "default_limit_gb": MEM_LIMIT_GB,
               # a block may be left unmeasured only if it says WHY (MEM_NOT_MEASURED)
               "blocks_not_run": sorted(study.get("not_run") or {}),
               "passed": bool(peaks and not unintended
                              and probe.get("growth_gb_per_solve") is not None
                              and probe_jit.get("growth_gb_per_solve") is not None
                              and (mem.get("probe_grad") or {}).get("growth_gb_per_solve")
                              is not None
                              and set(missing) <= set(mem.get("not_measured") or {}))}
    return g


# ----------------------------------------------------------------------------
# 9. figure

def figure(study: dict[str, Any], path: pathlib.Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    loop = study["loop"]
    evals = loop["evals"]
    acc_i = [e["i"] for e in evals if e["accepted"]]
    fig, axes = plt.subplots(3, 2, figsize=(12.5, 13.0))
    (ax0, ax1), (ax2, ax3), (ax4, ax5) = axes

    # (a) loss per evaluation
    ax0.plot([e["i"] for e in evals], [e["loss"] for e in evals], "o-", color="0.6", ms=4,
             lw=1.0, label="every objective evaluation")
    ax0.plot(acc_i, [evals[i]["loss"] for i in acc_i], "o", color="C0", ms=7,
             label="accepted iterate")
    sweep = study.get("sweep")
    if sweep and sweep.get("best_loss"):
        ax0.axhline(sweep["best_loss"]["loss"], color="C3", ls="--", lw=1.2,
                    label=f"best of the {sweep['n_forward_solves']} sweep solves")
    ax0.set_xlabel("objective evaluation (1 forward + 1 adjoint each)")
    ax0.set_ylabel("loss  = -Q/Q_ref + lam (dL/L_t)^2")
    ax0.set_title("(a) the loop")
    ax0.grid(alpha=0.3)
    ax0.legend(fontsize=8)

    # (b) L error and Q along the loop
    ax1.plot([e["i"] for e in evals], [100 * e["L_rel_error"] for e in evals], "o-",
             color="C0", ms=4, lw=1.0, label="L error [%]")
    ax1.axhspan(-100 * L_BAND, 100 * L_BAND, color="C0", alpha=0.12,
                label=f"+-{100 * L_BAND:g} % band")
    ax1.axhline(0.0, color="C0", lw=0.8, ls=":")
    ax1.set_xlabel("objective evaluation")
    ax1.set_ylabel("(L - L_target) / L_target  [%]", color="C0")
    axq = ax1.twinx()
    axq.plot([e["i"] for e in evals], [e["Q"] for e in evals], "s-", color="C1", ms=4, lw=1.0)
    axq.set_ylabel("Q_diff", color="C1")
    ax1.set_title("(b) constraint and objective")
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=8, loc="lower right")

    # (c) sweep vs loop in the (L error, Q) plane
    if sweep:
        ok = [p for p in sweep["points"] if p["feasible"]]
        ax2.scatter([100 * p["L_rel_error"] for p in ok], [p["Q"] for p in ok], s=28,
                    facecolor="none", edgecolor="C3",
                    label=f"{len(ok)} sweep forward solves")
    ax2.plot([100 * e["L_rel_error"] for e in evals], [e["Q"] for e in evals], "o-",
             color="0.6", ms=4, lw=0.8, label="loop evaluations")
    fin = loop["final"]
    if sweep and sweep.get("best_loss"):
        bl = sweep["best_loss"]
        ax2.plot([100 * bl["L_rel_error"]], [bl["Q"]], "s", color="C3", ms=9, mfc="none",
                 mew=2, label="best sweep point (loss)")
    ax2.plot([100 * fin["L_rel_error"]], [fin["Q"]], "*", color="C0", ms=18,
             label="gradient optimum")
    ax2.axvspan(-100 * L_BAND, 100 * L_BAND, color="C0", alpha=0.12,
                label=f"+-{100 * L_BAND:g} % L band")
    ax2.set_xlabel("(L - L_target) / L_target  [%]")
    ax2.set_ylabel("Q_diff")
    ax2.set_title("(c) sweep baseline vs the gradient loop")
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=8)

    # (d) gradient norm and the FD4 check
    bnds = loop.get("bounds_scaled", bounds_scaled())
    pg = [float(np.max(np.abs(projected_gradient(e["u"], e["grad"], bnds)))) for e in evals]
    ax3.semilogy([e["i"] for e in evals], [e["grad_inf"] for e in evals], "o-", color="0.6",
                 ms=4, lw=1.0, label="max |dloss/du| (full)")
    ax3.semilogy([e["i"] for e in evals], pg, "o-", color="C0", ms=4, lw=1.2,
                 label="max |projected grad| (what gtol tests)")
    if "gtol" in loop:
        ax3.axhline(loop["gtol"], color="C3", ls="--", lw=1.2, label="gtol (stop)")
    ax3.set_xlabel("objective evaluation")
    ax3.set_ylabel("max |grad| (scaled units)")
    title = "(d) the stop criterion"
    o4 = (study.get("gates") or {}).get("O4")
    if o4:
        title += f"  --  grad vs FD4: {o4['rel_to_grad_norm']:.1e} of |grad|"
    ax3.set_title(title)
    ax3.grid(alpha=0.3, which="both")
    ax3.legend(fontsize=8)
    fac = study.get("factorisations")
    if fac:
        sw = study.get("sweep") or {}
        fdb = study.get("fd_check") or {}
        loop_s = sum(e["seconds"] for e in evals)
        sweep_s = sum(pt["seconds"] for pt in sw.get("points", []))
        ax3.text(0.03, 0.06,
                 f"cost, measured at the optimum (N = {study['nominal']['grid']['n_unknowns']}):"
                 f"\none forward = {fac['forward']['misses']} LU factorisations = "
                 f"{fac['forward']['wall_seconds']:.0f} s "
                 f"({100 * fac['factor_over_wall_forward']:.0f} % of it inside _factor)\n"
                 f"the adjoint adds {fac['extra_factorisations_for_the_gradient']} "
                 f"factorisations (a cache-hit repeat is "
                 f"{100 * fac['cached_over_full']:.1f} % of a solve)\n"
                 f"FD4 at one point = 12 forward solves = {fdb.get('seconds', float('nan')):.0f} s"
                 f"\nloop {loop['n_objective_evaluations']} evaluations ({loop_s:.0f} s) vs "
                 f"sweep {sw.get('n_forward_solves', 0)} solves ({sweep_s:.0f} s)",
                 transform=ax3.transAxes, fontsize=7.5, va="bottom",
                 bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "0.7"})

    # (e) the memory measurement that sets the solve budget (section 0b)
    mem = study.get("memory") or {}
    probe = mem.get("probe") or {}
    if probe.get("footprint_gb"):
        xs = list(range(1, len(probe["footprint_gb"]) + 1))
        ax4.plot(xs, probe["footprint_gb"], "o-", color="C0", ms=5,
                 label="probe: physical footprint")
        ax4.plot(xs, probe["resident_gb"], "s--", color="C1", ms=4, lw=1.0,
                 label="probe: resident (RSS)")
        ax4.plot(xs, probe["maxrss_gb"], "^:", color="C2", ms=4, lw=1.0,
                 label="probe: ru_maxrss")
    for key, style, colour, label in (
            ("probe_jit", "o-", "C3", "probe, JITTED forward (the study's path)"),
            ("probe_grad", "D-", "C4", "probe, value_and_grad")):
        pr2 = mem.get(key) or {}
        if pr2.get("footprint_gb"):
            ax4.plot(range(1, len(pr2["footprint_gb"]) + 1), pr2["footprint_gb"], style,
                     color=colour, ms=5, lw=1.4, label=label)
    lp = [e.get("mem", {}).get("footprint_gb") for e in evals]
    if any(v is not None for v in lp):
        ax4.plot([i + 1 for i, v in enumerate(lp) if v is not None],
                 [v for v in lp if v is not None], "o-", ms=4, lw=1.0,
                 label="loop evaluations (1 per process)", color="C5")
    ax4.axhline(mem.get("limit_gb", MEM_LIMIT_GB), color="k", ls="--", lw=1.2,
                label=f"limit {mem.get('limit_gb', MEM_LIMIT_GB):g} GB")
    ax4.set_xlabel("solve within one process")
    ax4.set_ylabel("memory [GB]")
    ax4.set_title("(e) one process, repeated solves: the JITTED path grows")
    ax4.grid(alpha=0.3)
    ax4.legend(fontsize=7.5)

    # (f) the footprint each block actually reached
    blocks = mem.get("by_block") or {}
    if blocks:
        m1 = (study.get("gates") or {}).get("M1") or {}
        # the peak M1 reports (the max over the block's own logged notes), not
        # the end-of-chunk value, and the two kinds of overage kept apart
        peaks = m1.get("footprint_gb_by_block") or {}
        by_design = set(m1.get("blocks_over_limit_by_design") or {})
        names = list(blocks)
        vals = [float(peaks.get(n, blocks[n].get("footprint_gb_max",
                                                 blocks[n].get("footprint_gb_after", 0.0))))
                for n in names]
        lims = [float(blocks[n].get("limit_gb", MEM_LIMIT_GB)) for n in names]
        y = np.arange(len(names))
        ax5.barh(y, vals, color=["C1" if n in by_design else "C3" if v > q else "C0"
                                 for n, v, q in zip(names, vals, lims)])
        for i, q in enumerate(lims):
            ax5.plot([q], [i], "|", color="k", ms=14, mew=2)
        ax5.set_yticks(y)
        labels = []
        for n in names:
            k = blocks[n].get("solves_total", blocks[n].get("solves")) or 0
            tag = " (over BY DESIGN)" if n in by_design else f"  ({k} solves)" if k else ""
            labels.append(f"{n}{tag}")
        ax5.set_yticklabels(labels, fontsize=8)
        ax5.invert_yaxis()
        ax5.set_xlabel("peak physical footprint measured in the block [GB]")
        ax5.set_title("(f) per block, against the limit it ran under (|)")
        ax5.grid(alpha=0.3, axis="x")

    ref = study.get("refine")
    sub = (f"2-turn spiral, f0 = {FREQ / 1e9:g} GHz, Leontovich sigma = {SIGMA_CU:g} S/m "
           f"(delta/t = {study['nominal']['leontovich_validity']:.3f}, MARGINAL), "
           f"sigma_si = {SIGMA_SI:g} S/m, pad_cells = {PAD_CELLS}, "
           f"base_dx = W/1 (N = {study['nominal']['grid']['n_unknowns']})")
    if ref:
        sub += (f"\nW/2 re-solve of the optimum: L {100 * ref['points']['final']['L_shift']:+.1f} %, "
                f"Q {100 * ref['points']['final']['Q_shift']:+.1f} %  "
                f"-- absolute L is biased (see the module doc), the loop is not")
    fig.suptitle("FDFD spiral design loop D3 -- solve count and constraint accuracy\n" + sub,
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ----------------------------------------------------------------------------
# 10. main

def main(argv: Sequence[str] | None = None) -> int:
    """``_main`` under the memory limit: a trip is a clean stop (exit
    ``EXIT_MEM``) with everything measured so far already in the JSON, not a
    traceback and not an OOM kill."""
    try:
        return _main(argv)
    except MemBudget as exc:
        print(f"\nSTOPPED on the memory limit: {exc}", flush=True)
        dump = _MEM.get("dump")
        if callable(dump):
            dump()
        print("  rerun the remaining work with a smaller --max-solves; do NOT simply "
              "retry this chunk", flush=True)
        return EXIT_MEM


def _main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--maxiter", type=int, default=MAXITER)
    ap.add_argument("--no-sweep", action="store_true")
    ap.add_argument("--no-fd", action="store_true")
    ap.add_argument("--no-refine", action="store_true")
    ap.add_argument("--no-cost", action="store_true")
    ap.add_argument("--mode-cost", action="store_true",
                    help="run the four-execution-mode cost block; OFF by default because it "
                         "cannot be chunked and needs 12-25 GB (see BLOCKS_NOT_RUN)")
    ap.add_argument("--no-referee", action="store_true")
    ap.add_argument("--from-json", action="store_true",
                    help="reuse the blocks already in the JSON (and resume the partial ones)")
    ap.add_argument("--verify", action="store_true",
                    help="re-solve the nominal theta and compare with the JSON (drift check), "
                         "then exit; ~1 solve")
    ap.add_argument("--max-solves", type=int, default=None,
                    help="stop cleanly after this many three-fixture solves and exit "
                         f"{EXIT_INCOMPLETE} if work remains (memory control, see sections 0 and 0b)")
    ap.add_argument("--mem-limit", type=float, default=MEM_LIMIT_GB, metavar="GB",
                    help=f"physical-footprint limit of this process in GB "
                         f"(default {MEM_LIMIT_GB:g}); "
                         "raise it ONLY for a block measured to need more, and say so -- the "
                         "limit actually used is recorded in the JSON")
    ap.add_argument("--memory-probe-jit", type=int, default=0, metavar="K",
                    help="the same probe on the JITTED forward solve (the path the loop, the "
                         "sweep and the FD stencil take); K solves")
    ap.add_argument("--memory-probe-grad", type=int, default=0, metavar="K",
                    help="the same probe with jax.value_and_grad instead of a forward solve "
                         "(the number that sets the gradient loop's chunk size)")
    ap.add_argument("--memory-probe", type=int, default=0, metavar="K",
                    help="re-solve the nominal theta K times in THIS process, logging the "
                         "resident set after each (the study's memory measurement); K solves "
                         "on top of whatever else the chunk does")
    ap.add_argument("--recompute", nargs="*", default=(),
                    help="blocks NOT to reuse with --from-json")
    args = ap.parse_args(argv)

    import jax
    jax.config.update("jax_enable_x64", True)
    _BUDGET["used"], _BUDGET["max"] = 0, args.max_solves
    _MEM["blocks"], _MEM["probe"] = {}, None
    _MEM["probe_jit"], _MEM["probe_grad"] = None, None
    _MEM["limit_gb"] = float(args.mem_limit)

    t_start = time.time()
    previous: dict[str, Any] = json.loads(JSON_PATH.read_text()) if (
        args.from_json and JSON_PATH.exists()) else {}
    study: dict[str, Any] = {
        "fixture": {
            "freq": FREQ, "sigma_metal": SIGMA_CU, "sigma_si": SIGMA_SI, "n_turns": N_TURNS,
            "theta_nominal": list(THETA0), "params": list(PARAMS), "lead": LEAD, "margin": MARGIN,
            "base_dz": BASE_DZ, "metal_cells": METAL_CELLS, "pad_cells": PAD_CELLS,
            "pad_ratio": PAD_RATIO, "eps_si": EPS_SI, "eps_ox": EPS_OX,
            "stack": {"t_si": T_SI, "t_ox_low": T_OX_LOW, "t_m1": T_M1, "t_via": T_VIA,
                      "t_m2": T_M2, "t_ox_high": T_OX_HIGH, "t_air": T_AIR},
            "bounds": [list(b) for b in BOUNDS], "bounds_scaled": bounds_scaled(),
            "L_fraction": L_FRACTION, "L_band": L_BAND, "e_target": E_TARGET,
            "fd_step_rel": FD_STEP_REL, "sweep_n": SWEEP_N, "gtol_factor": GTOL_FACTOR,
        },
    }

    # a chunk that does not re-run the probe must not wipe the one that did
    _MEM["probe"] = (previous.get("memory") or {}).get("probe")
    _MEM["probe_jit"] = (previous.get("memory") or {}).get("probe_jit")
    _MEM["probe_grad"] = (previous.get("memory") or {}).get("probe_grad")
    # ... and neither must a chunk that skips a block (--no-sweep, --no-fd):
    # the per-block memory records are seeded from the JSON and only
    # OVERWRITTEN by a chunk that actually measures the block again. An
    # earlier revision dropped them, which silently deleted the sweep's and
    # the FD stencil's measured footprints from the JSON.
    _MEM["blocks"] = {k: dict(v, reused=True)
                      for k, v in ((previous.get("memory") or {}).get("by_block") or {}).items()}

    # A chunked run rewrites the whole JSON every time, so anything this
    # process neither recomputes nor visits (because an earlier block ran out
    # of solve budget) has to be carried across explicitly -- otherwise a
    # budget stop DELETES blocks that were already measured. "gates" is the
    # one thing not carried: it is derived, and only a complete run writes it.
    if args.from_json:
        for _k, _v in previous.items():
            if _k not in ("gates", "memory", "seconds", "seconds_by_block",
                          "seconds_all_blocks") and _k not in args.recompute:
                study.setdefault(_k, _v)

    def dump() -> None:
        # "seconds" is THIS process invocation; the study is run as resumable
        # chunks (see the module doc), so the total is the sum over the blocks
        study["memory"] = {
            "limit_gb": float(_MEM["limit_gb"]), "limit_gb_default": MEM_LIMIT_GB,
            "process_footprint_gb": footprint_gb(), "process_maxrss_gb": maxrss_gb(),
            "process_solves": int(_BUDGET["used"]),
            "by_block": _MEM["blocks"], "probe": _MEM["probe"],
            "probe_jit": _MEM.get("probe_jit"), "probe_grad": _MEM.get("probe_grad"),
            "not_measured": {k: v for k, v in MEM_NOT_MEASURED.items() if k in study},
            "note": "peak_gb is ru_maxrss of the process that MEASURED the block (monotone "
                    "over a process, so a block's own peak is peak_gb_after minus the "
                    "process high-water mark it started from); blocks marked reused=true "
                    "carry the record of the chunk that measured them",
        }
        study["seconds"] = time.time() - t_start
        study["seconds_by_block"] = {k: v["seconds"] for k, v in study.items()
                                     if isinstance(v, dict) and "seconds" in v}
        study["seconds_all_blocks"] = float(sum(study["seconds_by_block"].values()))
        JSON_PATH.write_text(json.dumps(study, indent=1, sort_keys=False))

    _MEM["dump"] = dump                          # so main() can flush on an RSS stop

    incomplete: list[str] = []

    prev_mem: dict[str, Any] = (previous.get("memory") or {}).get("by_block") or {}

    def carry(name: str) -> None:
        """A block reused from the JSON keeps the memory record of the chunk
        that actually measured it (this process did no solving for it)."""
        if name in prev_mem:
            _MEM["blocks"][name] = dict(prev_mem[name], reused=True)

    def measured(name: str, peak0: float, used0: int) -> None:
        """A block can be measured across SEVERAL chunks (the solve budget
        stops it and the next process continues it), so keep one entry per
        chunk and report the max of their peaks."""
        this = {"footprint_gb_before": peak0, "footprint_gb_after": footprint_gb(),
                "maxrss_gb_after": maxrss_gb(), "solves": int(_BUDGET["used"]) - used0,
                "limit_gb": float(_MEM["limit_gb"])}
        prior = prev_mem.get(name) or {}
        # chunk records written by the EARLIER instrumentation of this study
        # (ru_maxrss only, keys peak_gb_*) are dropped rather than merged: the
        # two metrics are not comparable, and a block re-measured now reports
        # the footprint of the chunks that measured the footprint
        chunks = [dict(c) for c in prior.get("chunks", []) if "footprint_gb_after" in c]
        chunks.append(this)
        _MEM["blocks"][name] = dict(this, chunks=chunks, reused=False,
                                    n_chunks=len(chunks),
                                    footprint_gb_max=max(c["footprint_gb_after"]
                                                         for c in chunks),
                                    solves_total=sum(c["solves"] for c in chunks))

    def block(name: str, fn: Callable[[], Any]) -> Any:
        if args.from_json and name in previous and name not in args.recompute:
            study[name] = previous[name]
            carry(name)
            print(f"  {name}: reused from the JSON", flush=True)
        else:
            peak0, used0 = footprint_gb(), int(_BUDGET["used"])
            try:                                 # record the memory even on a trip
                study[name] = fn()
            finally:
                measured(name, peak0, used0)
                dump()
            mem_check(f"block:{name}")           # dumped first, then guarded
        dump()
        return study[name]

    def resumable(name: str, fn: Callable[[dict[str, Any]], Any]) -> Any:
        """A block that can be stopped by the solve budget and continued by
        the next process: it is reused only when its ``complete`` flag is
        set, and otherwise restarted FROM the partial record."""
        prev = previous.get(name) if (args.from_json and name not in args.recompute) else None
        if prev and prev.get("complete"):
            study[name] = prev
            carry(name)
            print(f"  {name}: reused from the JSON (complete)", flush=True)
            dump()
            return study[name]
        study[name] = dict(prev) if prev else {}
        peak0, used0 = footprint_gb(), int(_BUDGET["used"])
        try:                                     # record the memory even on a trip
            fn(study[name])
        finally:
            measured(name, peak0, used0)
            dump()
        mem_check(f"block:{name}")               # dumped first, then guarded
        if not study[name].get("complete"):
            incomplete.append(name)
            print(f"  {name}: INCOMPLETE (solve budget {_BUDGET['used']}/{_BUDGET['max']})",
                  flush=True)
        return study[name]

    if args.verify:
        # script-only drift check (one 40 s solve, too slow for the test file):
        # the recorded nominal must still come out of the CURRENT
        # rfx.fdfd.spiral. Measured 2026-09-14 against a concurrently edited
        # spiral.py (the dut_kind="bar" addition): bit-identical, 0.0e0
        # relative on both L and Q.
        stored = json.loads(JSON_PATH.read_text())["nominal"]
        fresh = nominal_metrics(build_level(1.0))
        for key in ("L_diff", "Q_diff"):
            rel = abs(fresh[key] / stored[key] - 1.0)
            print(f"  {key}: fresh {fresh[key]:.12g}  stored {stored[key]:.12g}  "
                  f"rel {rel:.3e}", flush=True)
            if rel > 1e-9:
                print(f"  DRIFT in {key}: rerun the study", flush=True)
                return 1
        print("  the JSON reproduces on the current rfx.fdfd.spiral", flush=True)
        return 0

    print(f"design fixture: f0 = {FREQ:g} Hz, theta0 = "
          f"({R_OUT * 1e6:g}, {SPACING * 1e6:g}, {WIDTH * 1e6:g}) um, box "
          f"{[(lo * 1e6, hi * 1e6) for lo, hi in BOUNDS]} um", flush=True)
    model = build_level(1.0)
    print(f"  W/1 grid {model.shape}, N = {model.n_unknowns}", flush=True)

    def _nominal() -> dict[str, Any]:
        spend()
        return nominal_metrics(model)

    nom = block("nominal", _nominal)
    print(f"  nominal: L = {nom['L_diff'] * 1e12:.3f} pH, Q = {nom['Q_diff']:.4f}, "
          f"Leontovich delta/t = {nom['leontovich_validity']:.4f} "
          f"({'MARGINAL' if nom['leontovich_validity'] > 0.3 else 'ok'}), "
          f"forward solve {nom['forward_seconds']:.1f} s", flush=True)

    l_target = L_FRACTION * nom["L_diff"]
    q_ref = nom["Q_diff"]
    study["design"] = {"L_target": l_target, "Q_ref": q_ref, "L_fraction": L_FRACTION}
    dump()
    print(f"  L_target = {l_target * 1e12:.3f} pH ({100 * L_FRACTION:g} % of nominal), "
          f"Q_ref = {q_ref:.4f}", flush=True)

    if args.memory_probe:
        print(f"memory probe ({args.memory_probe} repeated solves in THIS process):", flush=True)
        peak0, used0 = footprint_gb(), int(_BUDGET["used"])
        spend(args.memory_probe)
        _MEM["probe"] = memory_probe(model, args.memory_probe)
        measured("memory_probe", peak0, used0)
        pr = _MEM["probe"]
        print(f"  footprint {pr['footprint_gb_first_last'][0]:.2f} -> "
              f"{pr['footprint_gb_first_last'][1]:.2f} GB over {pr['n_solves']} solves, "
              f"growth {pr['growth_gb_per_solve']:+.3f} GB/solve (resident "
              f"{pr['growth_gb_per_solve_resident']:+.3f}); at the {_MEM['limit_gb']:.1f} GB "
              f"limit that is {pr['solves_per_process_at_limit']} solves per process",
              flush=True)
        dump()

    if args.memory_probe_jit:
        print(f"memory probe, JITTED forward ({args.memory_probe_jit} solves in THIS process):",
              flush=True)
        peak0, used0 = footprint_gb(), int(_BUDGET["used"])
        spend(args.memory_probe_jit)
        _MEM["probe_jit"] = memory_probe(model, args.memory_probe_jit, mode="jit")
        measured("memory_probe_jit", peak0, used0)
        pj = _MEM["probe_jit"]
        print(f"  footprint {pj['footprint_gb_first_last'][0]:.2f} -> "
              f"{pj['footprint_gb_first_last'][1]:.2f} GB over {pj['n_solves']} solves, "
              f"growth {pj['growth_gb_per_solve']:+.3f} GB/solve (resident "
              f"{pj['growth_gb_per_solve_resident']:+.3f}); at the {MEM_LIMIT_GB:.0f} GB rule "
              f"that is {pj['solves_per_process_at_default_limit']} solves per process",
              flush=True)
        dump()

    if args.memory_probe_grad:
        print(f"memory probe, REVERSE MODE ({args.memory_probe_grad} value_and_grad passes "
              f"in THIS process):", flush=True)
        peak0, used0 = footprint_gb(), int(_BUDGET["used"])
        spend(args.memory_probe_grad)
        _MEM["probe_grad"] = memory_probe(model, args.memory_probe_grad, mode="grad")
        measured("memory_probe_grad", peak0, used0)
        pg = _MEM["probe_grad"]
        print(f"  footprint {pg['footprint_gb_first_last'][0]:.2f} -> "
              f"{pg['footprint_gb_first_last'][1]:.2f} GB over {pg['n_solves']} passes, "
              f"growth {pg['growth_gb_per_solve']:+.3f} GB/pass; at the "
              f"{_MEM['limit_gb']:.1f} GB limit that is "
              f"{pg['solves_per_process_at_limit']} passes per process", flush=True)
        dump()

    print("lambda (from measured gradients):", flush=True)
    def _lambda() -> dict[str, Any]:
        spend()
        return choose_lambda(model, l_target, q_ref)

    lam_rec = block("lambda", _lambda)
    lam = lam_rec["lam"]

    print("gradient loop (L-BFGS-B, jac from jax.value_and_grad):", flush=True)
    # resumable by REPLAY of its own evaluation log (see run_loop): the
    # process footprint grows per solve, so a 16-evaluation loop does not fit
    # in one process under the memory limit of section 0b
    resumable("loop", lambda rec: run_loop(model, l_target, q_ref, lam, maxiter=args.maxiter,
                                           dump=dump, record=rec, resume=dict(rec)))
    loop = study["loop"]
    if loop.get("complete"):
        print(f"  final theta = ({loop['final_theta'][0] * 1e6:.4f}, "
              f"{loop['final_theta'][1] * 1e6:.4f}, {loop['final_theta'][2] * 1e6:.4f}) um  "
              f"L = {loop['final_L'] * 1e12:.3f} pH ({100 * loop['final_L_rel_error']:+.3f} %)  "
              f"Q = {loop['final_Q']:.4f}", flush=True)

    if not args.no_sweep and not incomplete:
        print(f"sweep baseline ({SWEEP_N}^3 forward solves):", flush=True)
        resumable("sweep", lambda rec: run_sweep(model, l_target, q_ref, lam, dump=dump,
                                                 record=rec, resume=rec.get("points")))

    if not args.no_fd and not incomplete:
        print("O4: jax.grad vs FD4 at the final theta:", flush=True)
        resumable("fd_check", lambda rec: fd_check(model, loop["final_u"], l_target, q_ref, lam,
                                                   record=rec))

    def unbudgeted(name: str, n_solves: int, fn: Callable[[], Any]) -> bool:
        """A block that is NOT resumable (it measures within one process) and
        therefore cannot be split by the solve budget: if this chunk has
        already spent solves, leave it for a fresh process instead of piling
        its footprint on top (see section 0b)."""
        reuse = (args.from_json and name in previous and name not in args.recompute)
        if reuse:
            block(name, fn)
            return True
        if exhausted() or _BUDGET["used"] > 0:
            incomplete.append(name)
            print(f"  {name}: deferred to a fresh process ({_BUDGET['used']} solves already "
                  f"spent here)", flush=True)
            return False
        spend(n_solves)
        block(name, fn)
        return True

    if args.mode_cost and not args.no_cost and not incomplete:
        print("cost of one gradient (measured, LU cache cleared before each):", flush=True)
        unbudgeted("mode_cost", 7,
                   lambda: mode_cost(model, l_target, q_ref, lam, u=loop["final_u"]))
    elif "mode_cost" not in study:
        study["not_run"] = {k: v for k, v in BLOCKS_NOT_RUN.items()}

    if not args.no_cost and not incomplete:
        print("factorisation accounting (counts, not seconds):", flush=True)
        unbudgeted("factorisations", 2,
                   lambda: factorisation_accounting(model, l_target, q_ref, lam,
                                                    u=loop["final_u"]))

    if not args.no_referee and not incomplete:
        print("referee cross-check of the absolute bias at this geometry:", flush=True)
        unbudgeted("referee_bias", 1, lambda: referee_bias())

    if not args.no_refine and not incomplete:
        print("O5: the W/2 re-solve (report only):", flush=True)
        coarse = {"nominal": {"L": nom["L_diff"], "Q": nom["Q_diff"]},
                  "final": {"L": loop["final_L"], "Q": loop["final_Q"]}}
        resumable("refine", lambda rec: refine(loop["final_theta"], l_target, q_ref, lam, coarse,
                                               record=rec))

    if incomplete:
        dump()
        print(f"\nINCOMPLETE: {', '.join(incomplete)} -- rerun with --from-json "
              f"(solve budget {_BUDGET['used']}/{_BUDGET['max']})", flush=True)
        return EXIT_INCOMPLETE

    study["gates"] = evaluate_gates(study)
    dump()
    figure(study, PNG_PATH)
    print(f"\nwrote {JSON_PATH} and {PNG_PATH} ({study['seconds'] / 60:.1f} min)", flush=True)
    for name, gate in study["gates"].items():
        print(f"  {name}: passed={gate.get('passed')}  {gate['name']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
