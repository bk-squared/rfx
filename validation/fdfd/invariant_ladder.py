"""Study P: a level-invariant spiral fixture, and the joint GPU convergence
ladder that answers the V1 question of ``spiral_convergence.py``.

WHY. The old ladder (``spiral_convergence.json`` W/1..W/3 on the CPU,
``gpu_scaling.json`` adding W/4, W/6 through cuDSS) PLATEAUS at -9.8 / -9.3 /
-9.5 % below the bridge-corrected referee (333.055 pH) at W/3 / W/4 / W/6
instead of converging. Three things of that fixture are defined in CELLS.
Two are physical objects that change with the level: (a) the side walls and
the lid sit ``pad_cells`` graded cells beyond the last meshed cell and move
IN as the grid refines; (b) the short standard's grounded post spans
``[inner.i1 + 1, outer.i0 - 1)``, one CELL from each column, so the post and
the bridge arms widen with the level (the de-embedded quantity changes).
The third is a discretisation that is never refined: (c) the vertical grid
is frozen (``base_dz`` = 10 um at every level; no object moves -- the port
gap is one 10 um cell at every level). And the levels are not nested, so
Richardson across them is not formally valid. Gate P6 measures each of the
three alone: the plateau is mainly (c).

THE FIXTURE (additive ``SpiralSpec`` options of ``rfx.fdfd.spiral``; the
default build is untouched and the committed studies' tests still pass):
walls at the DUT bounding box -+ ``wall_margin_m`` and the lid
``lid_height_m`` above the ground, reached by a grading of ratio <= 1.5 whose
last line is pinned to the wall (``pad_to``); the post at a physical gap of
``SHORT_GAP`` = W from each column (x in [22, 32] um -- the old fixture's W/1
post, so level 1 differs from the old W/1 only in its walls), post, arm and
column edges forced as grid lines; the port gap a physical 10 um (a mandatory
z line); and NESTED JOINT refinement: level m is the level-1 mesh with every
interval -- x, y, z, the metal slabs and the padding included -- cut into m
equal cells (``refine``). The traced body-fitted stretch is unchanged, so
``jax.grad`` works at every level. Protocol: the small spiral of the old
study (2 turns, r_out 52 um, W = S = 10 um, t = 2 um, vacuum dielectrics),
100 MHz, uniform-current volumetric metal at 2e6 S/m (skin depth 35.6 um =
3.56 W = 17.8 t), level 1 with ``base_dz`` 10 um and ONE cell across each
metal slab (``METAL_CELLS``; the metal_cells-2 family is run as a
cross-check).

RESULTS (``invariant_ladder.json``; GPU runs in ``runs``)
--------------------------------------------------------
P1  fixture invariance, read back from the BUILT cell masks and port boxes
    at levels 1, 2, 3, 4, 6 (N = 18643 / 140990 / 466833 / 1095964 /
    3663258): every wall, lid, post, bridge-arm, column, port and interface
    coordinate identical to 0 m (tolerance 1e-12 m), the three fixtures'
    metal volumes to 2.2e-16, every level-1 line a line of every finer level
    (distance 0 m). The same read-back on the OLD fixture moves its walls by
    81.25 um and its short-standard volume by 54.6 % between W/1 and W/3.
    PASSES.
P2  the ladder converges monotonically: L_dut = 259.223 / 298.525 / 308.631 /
    312.675 pH at m = 1 / 2 / 3 / 4 (cells across W), steps +39.30 / +10.11 /
    +4.04 pH, observed order 1.627 over the finest three; Richardson (finest
    pair p = 1 / p = 2, observed order) 324.806 / 317.874 / 319.451 pH, range
    [317.874, 324.806] pH. PASSES.
P3  THE V1 ANSWER: against the referee 333.055 pH (area-exact Greenhouse +
    ground image minus the physical bridge; ONE number for every level) the
    per-level gaps are -22.17 / -10.37 / -7.33 / -6.12 % and the extrapolated
    range is -4.558 % to -2.477 %: INSIDE +-5 % (PASSES), NOT inside +-3 %
    (the p = 2 end is -4.56 %; only the p = 1 end is within 3 %). The same
    ladder in the RC twin protocol (below) extrapolates to [318.171,
    325.104] pH, and the independent metal_cells-2 family (levels 1-3:
    262.559 / 299.490 / 308.960 pH, observed order 1.452) to [316.535,
    327.899] pH -- the two level-1 meshes agree on the limit.
    Is [p = 2, p = 1] a bracket? (``extrapolation_checks``; reported, no
    gate.) The observed order RISES with refinement, 1.447 over levels 1-3
    to 1.627 over levels 2-4, so the coarse levels are pre-asymptotic (an
    error a h + b h^2 of one sign would have its observed order fall toward
    1). But the pair estimates close in from both sides -- p = 1: 337.827 /
    328.842 / 324.806 pH, p = 2: 311.626 / 316.715 / 317.874 pH over the
    pairs (1,2) / (2,3) / (3,4) -- and two other fits land inside the range:
    least squares L_inf + a h + b h^2 over the four levels 323.273 pH
    (-2.937 %) and the exact cubic through them 319.577 pH (-4.047 %). The
    p = 2 end (-4.558 %) is 0.44 % from the 5 % gate.
    Stated separately, not applied (``corrections``): the 10 W walls
    +0.186 % (the wall ladder), the metal resistance's RC contamination
    +0.095 % at m = 4 (the twin), the post short -0.356 % to -0.332 % (the
    thru probe 1.595 / 1.388 / 1.293 / 1.246 pH at m = 1-4, extrapolated to
    [1.105, 1.186] pH); all three together move the range to -4.646 % to
    -2.535 %. The referee's own conventions: the M2 -> M1 transition anywhere
    across the 10 um post 0.038 %, the underpass one 2 um layer up / down
    -0.521 % / +0.445 %, the omitted via (z-directed, no mutual with any x/y
    bar; its own partial self-inductance plus ground image) at most 0.034 %.
P4  ``jax.grad`` of L_dut vs FD4 (1 % step) of dL/dwidth through cuDSS at the
    finest level, m = 4: -2.45700821e-05 vs -2.45700688e-05 H/m, 5.45e-07
    relative (1.72e-07 at m = 3, from the harvested lane record
    validation/vessl/runs/fdfd-gpu-p1-20260915T150027Z/p1_ladder.json, not
    this JSON). PASSES (<= 1e-4).
P5  dL/dwidth = -20.49 / -22.51 / -23.84 / -24.57 pH/um, level-to-level
    changes 2.02 / 1.34 / 0.73 pH/um: decreasing. PASSES.
P6  the old fixture with ONE level-dependence switched at a time (the old
    fixture at the same level is the reference; L_case / L_old - 1): at W/1
    (a) physical walls +0.099 %, (b) physical short standard -1.6e-09 and
    (c) vertical grid refined +3.5e-10 -- the last two are identities at
    level 1 by construction, asserted to 1e-8; at W/2 (a) +0.283 %,
    (b) -1.043 %, (c) +2.777 %, all three +1.854 % (interaction -0.164 %),
    and the new fixture +2.258 % (the rest, +0.397 %, is the nested
    in-plane and padding mesh). PASSES. Read back from the built masks at
    W/2 and W/3 (``p6.objects``), (a) moves the walls (up to 129.06 um at
    W/2) and (b) the post and both bridge arms (5.00 um; short-standard
    volume 29.1 %), while (c) moves NO object (6.8e-21 m, volumes to
    7.8e-16): it only adds z lines (16 -> 31 at W/2). So the old plateau
    was mainly a DISCRETISATION deficiency -- the vertical grid never
    refined -- and not the fixture changing with the level: at W/2 the
    frozen z grid alone puts the old L 2.70 % below the same fixture with
    z refined (L_old / L_case - 1), while the two genuine fixture changes
    put it +1.05 % (the widened post) and -0.28 % (the closer walls) off;
    switched together, (a) and (b) change L by -0.760 %, offsetting part of
    (c)'s +2.777 %.

H1  cuDSS HYBRID (host + device) MEMORY WITH AN EXPLICIT DEVICE LIMIT -- the
    option the failed m = 5 attempt lacked (``rfx/fdfd/_cudss.py``,
    ``RFX_FDFD_CUDSS_HYBRID_LIMIT``; lane
    ``validation/vessl/lane_p3_hybrid.py``, block ``hybrid`` of the JSON,
    run 369367261289 on one RTX A6000). At m = 3 (N = 466833, an 11.37 GB
    factor) with a 6 GiB limit, against the in-memory run of the SAME level
    in the SAME job.

    The limit BINDS, and through the shipped code path: the limit reaches
    cuDSS at ``DirectSolver`` construction and nowhere else (the plan record
    carries ``hybrid_limit_applied = "constructor"``, which is what
    ``limit_binds_the_plan`` requires before it reads a plan as evidence),
    and cuDSS's plan for the same matrix then asks for 6.00 GB of device
    memory instead of 11.37 GB -- 1.86e-09 relative below the limit, i.e.
    the limit, against the gate's 1e-6 predicate -- and for 18.79 GB of host
    memory instead of 0.04 GB, while the process's peak resident set grows by
    7.90 GB (13.55 -> 21.45 GB). It costs 1.27x on the factorisation
    (23.1 -> 29.5 s), 2.26x on a triangular solve (0.58 -> 1.30 s) and 1.25x
    on the whole three-fixture value_and_grad (161.1 -> 200.7 s).

    The ANSWER: L_dut agrees to 9.00e-10 relative (308.630703 pH both ways)
    and the gradient to 1.90e-09 as a vector (worst component 2.22e-09).
    FAILS the 1e-9 the gate asks of the gradient -- and every control says
    the same thing about what 1e-9 means here, because cuDSS is not
    bit-reproducible across calls (one gradient is six factorisations and
    twelve triangular solves):

    * the same in-memory value_and_grad run TWICE in the same job differs by
      1.70e-09 in the gradient vector and 6.00e-10 in L, and its worst
      gradient component (2.38e-09) is LARGER than hybrid mode's (2.22e-09).
      Hybrid mode is 1.12x that control on the gradient vector -- and the
      CONTROL ITSELF is above 1e-9, so this gate's bound is one the solver
      does not meet against itself in-core: no memory mode could pass it;
    * the ladder's OWN level 3, measured on the RTX 4090 in another job,
      sits 5.37e-09 (gradient) and 1.50e-10 (L) from this job's in-memory
      run -- the same computation on a different card is FURTHER from it
      than hybrid mode is;
    * the SOLUTION VECTOR of the one-fixture probe: hybrid differs from
      in-core by 8.61e-05 (max-norm, scaled by max|x|) and in-core from
      itself by 6.70e-05, i.e. the same size. L_dut, a functional of x,
      agrees to 9e-10 while x itself agrees to 1e-4: the operator is
      ill-conditioned and the difference is the solver's, not the memory
      mode's (residuals 1.418e-08 in-core, 1.411e-08 hybrid);
    * and the reported difference is not itself reproducible to better than
      3.20x: the same comparison in the three P3 jobs gives 1.90e-09 /
      6.07e-09 for the gradient vector and 1.62e-09 / 1.12e-08 / 2.22e-09
      for its worst component (``hybrid.h1_replicates``). Every one of those
      is above 1e-9 and every one is within a small factor of that job's own
      in-core control, so the honest statement is "hybrid mode changes the
      gradient by about as much as running the same thing twice does", NOT
      "hybrid mode reproduces the gradient to 1e-9". The gate is left at
      1e-9 and reported FAILING; it is not widened.

    (What says the gradient itself is right is P4: AD against FD4 at
    5.45e-07.)
H2  m = 5 (N = 2128175) CONTINUES THE LADDER: NOT MEASURED, and the blocker
    is HOST memory, not the card. cuDSS hybrid mode keeps the whole factor
    in host memory ("Factors L and U ... must fit into the host memory"),
    and its plan for this level asks for 97.60 GB of host memory (in-core
    factor 91.87 GB) while needing 8.75 GB of device memory at the minimum
    (that is the hybrid plan's figure; the in-core plan of the same matrix
    says 16.78 GB -- both are far below the card, which is the point, and
    neither enters the feasibility test). The gpu-a6000-1 preset's container
    may hold 32.0 GB (cgroup v2 ``memory.max`` = 34359738368; ``free -g``
    reports the NODE's 251.57 GB, which is why the lane reads the cgroup),
    and 15.24 GB of that was already the assembled operator: 97.60 + 15.24 +
    4 GB of margin against 32.0 GB, so the lane wrote the numbers and stopped
    instead of being OOM-killed. That ceiling -- 12.76 GB of factor here --
    is BELOW the card's own 47.54 GB, so on this preset hybrid mode cannot
    reach past the device at all. No preset of this cluster fixes it by host
    memory alone (gpu-a6000-1-hi-mem is 64 GiB, still under 97.60; the
    230 GiB preset takes three GPUs). The only single-GPU route is a bigger
    CARD: the gpu-h200 preset's 141 GB would hold the 91.87 GB factor in
    device memory with no hybrid mode at all -- untested here, another GPU
    type, and not one of this study's presets. The ladder stays m = 1-4: P2,
    P3 and P5 are exactly as recorded above.

Supporting measurements
-----------------------
* Wall distance (level 2, metal_cells 2): walls and lid at 10 / 20 / 40 /
  80 W give 299.490 / 300.037 / 300.018 / 300.047 pH (+0.183 / -0.006 /
  +0.009 % per doubling). Rule: the smallest margin whose change to the
  doubled one is below 0.2 % -- 10 W; the 40 -> 80 W change is 0.009 %. The
  10 W residual (+0.186 % to 80 W) is carried as a stated correction; 20 W
  would have cost 1.35x the unknowns (P0's plans: level 3 at 20 W is
  23.45 GB of cuDSS factor against 15.65 GB at 10 W, metal_cells 2).
* Uniform-current validity, measured at level 3 (3 cells across W,
  metal_cells 2, N = 581430): L_dut = 296.937 / 308.129 / 308.960 / 309.113 /
  309.225 / 309.232 pH at 3e5 / 1e6 / 2e6 / 3e6 / 1e7 / 3e7 S/m (skin depth
  9.19 / 5.03 / 3.56 / 2.91 / 1.59 / 0.92 W). No current crowding shows even
  at delta < W; what bends the curve at LOW sigma is the RC contamination of
  Im Z / omega (~ C R^2, Re Z_diff = 12.8 Ohm at 2e6 S/m). The RC twin
  proves it: 10 MHz with 2e7 S/m -- the same skin depth, a tenth of the
  resistance -- gives 309.233 pH, +0.089 % over 100 MHz / 2e6 and equal to
  the 3e7 point to 6e-6.
* Cost through cuDSS (complex128, RTX 4090 for m <= 3, RTX A6000 48 GB for
  m = 4): one factorisation 11.1 s at m = 3 with 12.65 GB of device memory
  IN USE (factor, operands and cupy's pool; the factor itself is cuDSS's
  11.37 GB plan estimate, which is what gate H1's limit is set below) and
  120.2 s at m = 4 with 38.82 GB (plan: 36.61 GB permanent); value_and_grad
  of the three-fixture L_dut 6.9 / 15.0 / 76.5 / 755.3 s at m = 1-4. The
  joint refinement is truly 3-D (N = 18643 -> 1095964 from m = 1 to 4, the
  padding refined too), so the factor grows much faster than the old in-plane
  ladder's: the plans give 91.87 GB at m = 5 and 193.78 GB at m = 6 -- m = 6
  and m = 8 do not fit any card here. The m = 5 attempt of run 369367261252
  (cuDSS hybrid memory, no device limit) died with ALLOC_FAILED at 47.52 of
  47.54 GB in use; hybrid mode now takes an explicit device limit and moves
  the factor off the card as asked (gate H1, which nevertheless fails on the
  gradient's 1e-9), and the reason m = 5 is still not run is HOST memory
  (gate H2).
  The ladder stops at m = 4.
* Gradient memory: a factor cache of 1 keeps the last fixture's A factor
  alive while cuDSS factorises its A^T (it has no transposed solve), so the
  first m = 3 run died with ALLOC_FAILED on the 24 GB card; a cache of 0
  (one factor resident) costs nothing on cuDSS and is what every level >= 3
  ran with.

WHAT A 3 % ABSOLUTE L NEEDS. From the fitted error model of the finest three
levels (``resolution_for_3_percent``) the discretisation error at m = 4 is
3.74 % at p = 1, 2.12 % at the observed order and 1.64 % at p = 2: m = 4
is under 3 % only if the order is at least the observed one, NOT at
p = 1. A 3 % discretisation error needs 2.95 (p = 2) / 3.23 (observed) /
4.98 (p = 1) cells across W, so the first level of THIS fixture under 3 %
at every order in [1, 2] is m = 5 (plan: 91.87 GB of factor; not run).
And the limit itself sits 2.48 % to 4.56 % below the referee, so 3 %
ABSOLUTE is not bought by resolution alone: at the p = 2 end no mesh of
this fixture reaches it.

THE RESIDUAL IS NOT EXPLAINED (``residual``). Every model difference
between the fixture and the referee that this study measured -- the walls
+0.186 %, the RC contamination +0.095 %, the post short -0.356 %, the
omitted via <= 0.034 %, the transition anywhere across the post <= 0.038 %
-- adds up, at its worst and all pulling one way, to 0.709 %; applying the
signed ones moves the range only to -4.646 % to -2.535 %, which leaves at
least 2.462 % that nothing measured here accounts for. Every finest-level
estimate (the finest pair at p = 1 and p = 2, the observed order, the two
fits) puts the limit at least 2.477 % below the referee. But the residual
is the size of the extrapolation's own p-spread (2.081 %) and the ladder is
pre-asymptotic (its observed order still rising), so four levels cannot
tell an unidentified model difference from an extrapolation error: it is
reported as unexplained, not attributed.

SECONDARY: THE PAPER GEOMETRY (``paper``; reported, no gate). Study H's
square spiral (3 turns, r_out 218 um, W 30 um, S 14 um, M2 3 um,
``rfic_spiral.py``'s stack with vacuum dielectrics, ground 131.54 um below
the M2 centre line) on the same invariant fixture (10 W walls, the post
30 um from each column, a 60 um port gap; level 1 has 2 cells across W and
one across each metal slab) with a VALID uniform-current protocol: 10 MHz,
2e6 S/m, skin depth 112.5 um = 3.75 W = 37.5 t. Its plateau at level 2
(4 cells across W, N = 571252): 3238.45 / 3317.28 / 3336.99 / 3340.65 /
3343.34 / 3343.89 pH at 5e5 / 1e6 / 2e6 / 3e6 / 1e7 / 3e7 S/m (skin depth
7.50 ... 0.97 W) and the 1 MHz / 2e7 S/m twin 3343.49 pH (+0.195 %): RC
again, no crowding -- at 0.97 W, the skin depth the earlier GPU lane had
(3e6 S/m at 100 MHz), L sits 0.012 % ABOVE the twin, so current crowding
is NOT what made that lane's gap grow (-13.2 -> -15.8 %); the old
cell-defined build was (its walls, post and never-refined vertical grid,
not separated for this geometry). Levels 1 / 2 (N = 74554 / 571252): 3072.35 / 3336.99 pH
against its referee 3560.37 pH (``rfic_spiral.referee_deembedded``):
-13.71 % / -6.27 %, the gap now SHRINKS with refinement. Level 3 is
N = 1899954 with a 65.79 GB factor (plan) and is not run, so only a
two-point extrapolation exists: 3601.64 (p = 1) / 3425.21 pH (p = 2) =
+1.16 % / -3.80 %. The 10 W walls are not converged for this larger
geometry: 5 / 10 / 20 W give 2980.13 / 3072.35 / 3087.08 pH at level 1
(+0.48 % from 10 to 20 W), stated, not applied.

Run::

    .venv/bin/python validation/fdfd/invariant_ladder.py [--from-json] [--no-figure]

assembles ``invariant_ladder.json`` and ``invariant_ladder.png`` from the
harvested GPU lane JSONs (``validation/vessl/runs/fdfd-gpu-p*``, lanes
``validation/vessl/lane_p*.py``), recomputing the referee, gate P1 and P6's
object read-back (``p6.objects``) on the host; ``--from-json`` reuses those
three blocks. The run ledger is
``validation/vessl/runs/p_run_ledger.json``.

Figure: (a) L_dut against h / W for the primary family, the metal_cells-2
family and the old cell-defined ladder, the referee with +-3 / +-5 % bands
and the Richardson range; (b) gate P6's per-artefact changes at W/1 and W/2;
(c) dL/dwidth by level with the FD4 point; (d) the wall ladder; (e) the
uniform-current plateau with the RC twin; (f) device memory against N.
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
SC_PATH = HERE / "spiral_convergence.py"
SC_JSON = HERE / "spiral_convergence.json"
JSON_PATH = HERE / "invariant_ladder.json"
PNG_PATH = HERE / "invariant_ladder.png"
RUNS = REPO / "validation" / "vessl" / "runs"

# ---- the fixture: spiral_convergence.json's small spiral -------------------
FREQ = 1e8
N_TURNS = 2
R_OUT, SPACING, WIDTH = 52e-6, 10e-6, 10e-6
LEAD = 1.5 * WIDTH
MARGIN = 10e-6             # meshed box beyond the polygons (= the old study's)
T_SI, T_OX_LOW, T_M1, T_VIA, T_M2, T_OX_HIGH, T_AIR = 20e-6, 1e-6, 2e-6, 2e-6, 2e-6, 3e-6, 10e-6
Z_TOP = T_SI + T_OX_LOW + T_M1 + T_VIA + T_M2 + T_OX_HIGH + T_AIR      # 40 um
BASE_DZ = 10e-6            # LEVEL-1 vertical cell (then subdivided with the level)
METAL_CELLS = 1            # LEVEL-1 cells across each metal slab (the primary family;
                           # the P0 probe blocks and the cross-check family use 2)
METAL_CELLS_ALT = 2
OLD_METAL_CELLS = 2        # spiral_convergence.py's METAL_CELLS (the old fixture)
WALL_W = 10.0              # the chosen wall margin in widths (the wall ladder's rule)
PAD_RATIO = 1.5
SHORT_GAP = WIDTH          # physical isolating gap column -> post: post x in [22, 32] um,
                           # exactly the old fixture's W/1 post (so level 1 = old W/1 there)
PORT_GAP = 10e-6           # physical port gap: column bottom at z = 10 um (= old W/1)
SIGMA = 2e6                # uniform-current volumetric metal: delta = 35.6 um = 3.56 W = 17.8 t
FREQ_RC = 1e7              # the RC-isolating twin: SIGMA_RC at FREQ_RC has the same delta
SIGMA_RC = 2e7
PLATEAU_SIGMAS = (3e5, 1e6, 2e6, 3e6, 1e7, 3e7)
WALL_LADDER = (10.0, 20.0, 40.0, 80.0)     # wall margins in units of W (level 2)
WALL_TOL = 0.002           # the task's criterion: < 0.2 % between successive margins
GROUND_H = T_SI + T_OX_LOW + T_M1 + T_VIA + 0.5 * T_M2          # 26 um, ground -> M2 centre
DZ_UNDER = 0.5 * T_M1 + T_VIA + 0.5 * T_M2                      # 4 um, M2 centre -> M1 centre
FD_STEP_REL = 0.01
PARAMS = ("r_out", "spacing", "width")
THETA0 = (R_OUT, SPACING, WIDTH)
MU0 = 4e-7 * np.pi


def fd4(f: Callable[[float], float], x0: float, d: float) -> float:
    return (-f(x0 + 2 * d) + 8 * f(x0 + d) - 8 * f(x0 - d) + f(x0 - 2 * d)) / (12 * d)


def skin_depth(freq: float, sigma: float) -> float:
    return float(np.sqrt(2.0 / (2.0 * np.pi * freq * MU0 * sigma)))


def _load(path: pathlib.Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_sc() -> Any:
    """``spiral_convergence.py`` (the old study), unmodified: its referee
    construction and its Richardson are reused, not reinvented."""
    return _load(SC_PATH, "spiral_convergence_for_p")


# ----------------------------------------------------------------------------
# 1. fixtures

def stack(metal_cells: int = METAL_CELLS, base_dz: float = BASE_DZ):
    from rfx.fdfd import spiral as sm
    return sm.SmallStack(t_si=T_SI, t_ox_low=T_OX_LOW, t_m1=T_M1, t_via=T_VIA, t_m2=T_M2,
                         t_ox_high=T_OX_HIGH, t_air=T_AIR, base_dz=base_dz,
                         metal_cells=metal_cells, eps_si=1.0, eps_ox=1.0)


def spec_new(m: int, wall_w: float, metal_cells: int = METAL_CELLS, pad_refine: str = "uniform",
             dut_kind: str = "spiral", bar_length: float = 0.0):
    """The level-invariant fixture at level ``m``: walls ``wall_w`` widths
    from the DUT bounding box, the lid the same distance above the stack,
    the physical short standard and port gap, nested joint refinement."""
    from rfx.fdfd import spiral as sm
    wm = float(wall_w) * WIDTH
    return sm.SpiralSpec(n_turns=N_TURNS, r_out=R_OUT, spacing=SPACING, width=WIDTH, lead=LEAD,
                         base_dx=WIDTH, margin=MARGIN, stack=stack(metal_cells),
                         wall_margin_m=wm, lid_height_m=Z_TOP + wm, short_gap_m=SHORT_GAP,
                         port_gap_m=PORT_GAP, refine=int(m), pad_refine=pad_refine,
                         pad_ratio=PAD_RATIO, dut_kind=dut_kind, bar_length=bar_length)


def spec_old(div: int, walls: bool = False, short: bool = False, vertical: bool = False,
             wall_w: float = 20.0):
    """The OLD cell-defined fixture of ``spiral_convergence.py`` at
    ``base_dx = W / div`` (pad_cells 4, post one cell from each column,
    base_dz frozen), with any of the three level-dependences switched to the
    physical definition: ``walls`` (a), ``short`` (b), ``vertical`` (c: the
    level-1 z lines, lid pad included, subdivided ``div`` times with the
    port gap held at 10 um)."""
    from rfx.fdfd import spiral as sm
    kw: dict[str, Any] = dict(n_turns=N_TURNS, r_out=R_OUT, spacing=SPACING, width=WIDTH,
                              lead=LEAD, base_dx=WIDTH / div, margin=MARGIN,
                              stack=stack(OLD_METAL_CELLS),
                              port_gap_cells=1, pad_cells=4, pad_ratio=PAD_RATIO)
    if walls:
        wm = float(wall_w) * WIDTH
        kw.update(pad_cells=0, wall_margin_m=wm, lid_height_m=Z_TOP + wm)
    if short:
        kw.update(short_gap_m=SHORT_GAP)
    if vertical:
        kw.update(z_refine=int(div), port_gap_m=PORT_GAP)
    return sm.SpiralSpec(**kw)


def build(spec):
    from rfx.fdfd import spiral as sm
    return sm.build_spiral(spec)


def l_dut(model, theta=None, sigma: float | None = SIGMA, freq: float = FREQ):
    """De-embedded ``L_diff`` (H) of the volumetric-metal fixture (``sigma =
    None``: PEC)."""
    from rfx.fdfd import spiral as sm
    return sm.solve_spiral(model, freq, theta=theta, sigma_volumetric=sigma).L_diff


def model_record(model) -> dict[str, Any]:
    x, y, z = (np.asarray(v) for v in (model.x_nom, model.y_nom, model.z))
    return {"shape": list(model.shape), "n_unknowns": int(model.n_unknowns),
            "wall_x": [float(x[0]), float(x[-1])], "wall_y": [float(y[0]), float(y[-1])],
            "lid_z": float(z[-1]), "dx_min": float(np.diff(x).min()),
            "dz_min": float(np.diff(z).min()),
            "build_seconds": float(model.build_seconds)}


# ---- the SECONDARY geometry: study H's paper-derived square spiral ---------
RFIC_PATH = HERE / "rfic_spiral.py"
PAPER = {"n_turns": 3, "r_out": 218e-6, "spacing": 14e-6, "width": 30e-6, "lead": 45e-6,
         "margin": 30e-6, "t_si": 120e-6, "t_ox_low": 6.04e-6, "t_m1": 2e-6, "t_via": 2e-6,
         "t_m2": 3e-6, "t_ox_high": 1.5e-6, "t_air": 60e-6, "base_dz": 60e-6,
         "port_gap": 60e-6, "short_gap": 30e-6, "wall_w": 10.0}
# delta >= 3 W = 90 um and >= 10 t = 30 um: (10 MHz, 2e6 S/m) gives 112.5 um
# = 3.75 W; the RC twin (1 MHz, 2e7 S/m) has the same delta, 1/10 the resistance
PAPER_FREQ, PAPER_SIGMA = 1e7, 2e6
PAPER_FREQ_RC, PAPER_SIGMA_RC = 1e6, 2e7
PAPER_PLATEAU_SIGMAS = (5e5, 1e6, 2e6, 3e6, 1e7, 3e7)


def spec_paper(m: int, metal_cells: int = 1, wall_w: float | None = None):
    """Study H's paper-geometry square spiral (``rfic_spiral.py``'s stack,
    vacuum dielectrics) on the level-invariant fixture at level ``m``."""
    from rfx.fdfd import spiral as sm
    P = PAPER
    st = sm.SmallStack(t_si=P["t_si"], t_ox_low=P["t_ox_low"], t_m1=P["t_m1"], t_via=P["t_via"],
                       t_m2=P["t_m2"], t_ox_high=P["t_ox_high"], t_air=P["t_air"],
                       base_dz=P["base_dz"], metal_cells=metal_cells, eps_si=1.0, eps_ox=1.0)
    wm = float(P["wall_w"] if wall_w is None else wall_w) * P["width"]
    return sm.SpiralSpec(n_turns=P["n_turns"], r_out=P["r_out"], spacing=P["spacing"],
                         width=P["width"], lead=P["lead"], base_dx=P["width"], margin=P["margin"],
                         stack=st, wall_margin_m=wm, lid_height_m=st.z_top + wm,
                         short_gap_m=P["short_gap"], port_gap_m=P["port_gap"], refine=int(m),
                         pad_ratio=PAD_RATIO)


# ----------------------------------------------------------------------------
# 2. gate P1: the fixture's coordinates, read back from the BUILT masks

def _box(mask: np.ndarray, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> list[float]:
    ii, jj, kk = np.nonzero(mask)
    return [float(x[ii.min()]), float(x[ii.max() + 1]), float(y[jj.min()]),
            float(y[jj.max() + 1]), float(z[kk.min()]), float(z[kk.max() + 1])]


def _cell_volume(mask: np.ndarray, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    dv = np.diff(x)[:, None, None] * np.diff(y)[None, :, None] * np.diff(z)[None, None, :]
    return float(np.sum(dv[mask]))


def fixture_coordinates(model) -> dict[str, Any]:
    """Every physical object of the three fixtures, as coordinates measured
    from the built cell masks and port boxes (NOT from the spec): the five
    walls, the two lead columns (``open`` cells, split by x), the two ports
    (Ez box corners), the short standard's post (the ``short``-only cells that
    touch the ground) and its two bridge arms (the ``short``-only cells on
    each metal level), the DUT metal per layer, the stack interfaces present
    as z lines, and the metal volumes of the three fixtures."""
    from scipy import ndimage

    x, y, z = (np.asarray(v, dtype=np.float64) for v in (model.x_nom, model.y_nom, model.z))
    st = model.spec.stack
    op, sh, du = (np.asarray(model.cells[k], dtype=bool) for k in ("open", "short", "dut"))
    lab, n = ndimage.label(op)
    cols = sorted((_box(lab == i + 1, x, y, z) for i in range(n)), key=lambda b: b[0])
    extra = sh & ~op
    # the post: the short-only cells ON the ground, extended up as long as the
    # whole footprint stays metal (the arms join it from the side)
    post = None
    foot = extra[:, :, 0]
    if foot.any():
        full = np.array([bool(np.all(extra[:, :, k][foot])) for k in range(extra.shape[2])])
        ktop = int(np.argmin(full)) if not full.all() else extra.shape[2]
        post = np.zeros_like(extra)
        post[:, :, :ktop] = foot[:, :, None]
    zc = 0.5 * (z[:-1] + z[1:])
    k_m1 = (zc > st.z_m1) & (zc < st.z_m1 + st.t_m1)
    k_m2 = (zc > st.z_m2) & (zc < st.z_m2 + st.t_m2)
    k_via = (zc > st.z_via) & (zc < st.z_via + st.t_via)
    arm_m2 = extra & k_m2[None, None, :]
    arm_m1 = extra & k_m1[None, None, :]
    metal = du & ~op
    ports = []
    for p in model.ports:
        (i0, j0, k0), (i1, j1, k1) = p.lo, p.hi
        ports.append([float(x[i0]), float(x[i1 - 1]), float(y[j0]), float(y[j1 - 1]),
                      float(z[k0]), float(z[k1])])
    interfaces = {}
    for name, zz in (("si_top", st.t_si), ("m1_bottom", st.z_m1), ("via_bottom", st.z_via),
                     ("m2_bottom", st.z_m2), ("m2_top", st.z_m2 + st.t_m2),
                     ("ox_top", st.z_ox_top), ("stack_top", st.z_top)):
        d = np.abs(z - zz)
        interfaces[name] = float(z[int(np.argmin(d))]) if d.min() <= 1e-12 else None
    out: dict[str, Any] = {
        "walls": {"x_lo": float(x[0]), "x_hi": float(x[-1]), "y_lo": float(y[0]),
                  "y_hi": float(y[-1]), "ground": float(z[0]), "lid": float(z[-1])},
        "columns": {"inner": cols[0], "outer": cols[-1]} if len(cols) == 2 else {"n": n},
        "ports": {"outer": ports[0], "inner": ports[1]},
        "post": _box(post, x, y, z) if post is not None else None,
        "bridge_m2": _box(arm_m2, x, y, z) if arm_m2.any() else None,
        "bridge_m1": _box(arm_m1, x, y, z) if arm_m1.any() else None,
        "dut_m2": _box(metal & k_m2[None, None, :], x, y, z),
        "dut_m1": _box(metal & k_m1[None, None, :], x, y, z) if (metal & k_m1[None, None, :]).any() else None,
        "dut_via": _box(metal & k_via[None, None, :], x, y, z) if (metal & k_via[None, None, :]).any() else None,
        "interfaces": interfaces,
        "volume": {k: _cell_volume(np.asarray(model.cells[k], dtype=bool), x, y, z)
                   for k in ("dut", "open", "short")},
    }
    return out


def _flatten(d: Any, prefix: str = "") -> dict[str, float]:
    out: dict[str, float] = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out.update(_flatten(v, f"{prefix}{k}."))
    elif isinstance(d, (list, tuple)):
        for i, v in enumerate(d):
            out.update(_flatten(v, f"{prefix}{i}."))
    elif d is None:
        out[prefix[:-1]] = float("nan")
    else:
        out[prefix[:-1]] = float(d)
    return out


def p1_gate(models: dict[int, Any]) -> dict[str, Any]:
    """Gate P1 from built models: (i) every fixture coordinate identical to
    1e-12 m across the levels (the metal volumes to 1e-12 relative), (ii)
    every level-1 x, y and z line a line of every finer level (1e-12 m)."""
    levels = sorted(models)
    coords = {m: fixture_coordinates(models[m]) for m in levels}
    flat = {m: _flatten({k: v for k, v in coords[m].items() if k != "volume"}) for m in levels}
    ref = flat[levels[0]]
    worst_coord, worst_key = 0.0, ""
    missing = [k for k, v in ref.items() if not np.isfinite(v)]
    for m in levels[1:]:
        for k, v in ref.items():
            d = abs(flat[m].get(k, np.inf) - v) if np.isfinite(v) else 0.0
            if d > worst_coord:
                worst_coord, worst_key = d, k
    vol_rel = 0.0
    for m in levels[1:]:
        for k, v in coords[levels[0]]["volume"].items():
            vol_rel = max(vol_rel, abs(coords[m]["volume"][k] / v - 1.0))
    base = models[levels[0]]
    nest: dict[str, Any] = {}
    worst_nest = 0.0
    for m in levels[1:]:
        fine = models[m]
        rec = {}
        for ax, a, b in (("x", base.x_nom, fine.x_nom), ("y", base.y_nom, fine.y_nom),
                         ("z", base.z, fine.z)):
            a, b = np.asarray(a), np.asarray(b)
            d = np.abs(a[:, None] - b[None, :]).min(axis=1)
            rec[ax] = float(d.max())
            worst_nest = max(worst_nest, rec[ax])
        rec["shape"] = list(fine.shape)
        nest[str(m)] = rec
    tol = 1e-12
    return {"levels": levels, "coordinates": {str(m): coords[m] for m in levels},
            "worst_coordinate_deviation_m": worst_coord, "worst_coordinate_key": worst_key,
            "undefined_coordinates": missing,
            "worst_volume_rel_deviation": vol_rel,
            "nesting_worst_distance_m": worst_nest, "nesting": nest, "tolerance_m": tol,
            "passed": bool(worst_coord <= tol and worst_nest <= tol and vol_rel <= 1e-12
                           and not missing)}


# ----------------------------------------------------------------------------
# 3. the referee: ONE number for every level

def post_coordinates(theta: Sequence[float] = THETA0) -> tuple[float, float]:
    """x edges of the physical short-standard post: the inner lead column's
    ``+x`` edge plus the gap, the outer column's ``-x`` edge minus it."""
    r_out, spacing, width = (float(v) for v in theta)
    a0 = r_out - 0.5 * width
    a_in = a0 - (width + spacing) * N_TURNS
    return a_in + 0.5 * width + SHORT_GAP, a0 - 0.5 * width - SHORT_GAP


def referee_block() -> dict[str, Any]:
    """The Greenhouse referee of the level-invariant fixture.

    ``L(strip) - L(bridge)`` with the old study's construction
    (``spiral_convergence.referee_value`` / ``bridge_segments``: area-exact
    corners, ground image at 26 um, reference planes at the lead-column
    footprint centres, underpass a Greenhouse bar 4 um below the M2 centre
    line). What is new is that the bridge is ONE physical object at every
    level: an M2 arm from the outer plane to the post and an M1 arm from the
    post to the inner plane, the M2 -> M1 transition at the post CENTRE
    (split 0.5, x = 27 um), with the transition anywhere across the post's
    physical width [22, 32] um reported as the split band. Also reported:
    the via treatment (omitted: z-directed, it has no mutual with any x/y
    bar; its own partial self-inductance and its ground-image mutual bound
    the omission) and the underpass-depth sensitivity (the M1 bar one 2 um
    layer up / down, bridge's M1 arm moved with it)."""
    sc = load_sc()
    sg = sc.load_referee()
    x0, x1 = post_coordinates()
    xo, xi, _ = sc.terminal_points(THETA0)
    split_c = (0.5 * (x0 + x1) - xi) / (xo - xi)
    strip = sc.referee_value(sg, THETA0)
    bridge = sc.bridge_value(sg, THETA0, split=split_c)
    total = float(strip.total - bridge.total)
    band = {}
    for name, xs in (("post_inner_edge", x0), ("post_centre", 0.5 * (x0 + x1)), ("post_outer_edge", x1)):
        s = (xs - xi) / (xo - xi)
        band[name] = {"x_transition": xs, "split_on_m2": s,
                      "bridge": float(sc.bridge_value(sg, THETA0, split=s).total),
                      "deembedded": sc.referee_deembedded(sg, THETA0, split=s)}
    vals = [v["deembedded"] for v in band.values()]
    # underpass depth one layer up / down (the bridge's M1 arm moves with it)
    depth: dict[str, Any] = {}
    dz0 = sc.DZ_UNDER
    try:
        for name, dz in (("one_layer_up", dz0 - T_VIA), ("nominal", dz0), ("one_layer_down", dz0 + T_M1)):
            sc.DZ_UNDER = dz
            v = sc.referee_deembedded(sg, THETA0, split=split_c)
            depth[name] = {"dz_underpass": dz, "deembedded": v, "rel": v / total - 1.0}
    finally:
        sc.DZ_UNDER = dz0
    # the via: a 2 um tall, W x W column between M1 and M2 (z-directed)
    l_via = sg.self_inductance_bar(T_VIA, WIDTH, WIDTH)
    z_via_c = T_SI + T_OX_LOW + T_M1 + 0.5 * T_VIA
    # ground image of a VERTICAL current flows in the SAME direction: +M
    m_img = sg.mutual_inductance_bars(T_VIA, WIDTH, WIDTH, T_VIA, WIDTH, WIDTH, 0.0, 0.0,
                                      -2.0 * z_via_c)
    via = {"what": "omitted by the referee (z-directed; every mutual with the x/y bars "
                   "vanishes); its own partial self-inductance plus its ground-image mutual "
                   "bound what the omission costs",
           "self": float(l_via), "image_mutual": float(m_img),
           "bound": float(l_via + m_img), "bound_rel": float((l_via + m_img) / total)}
    dee_old = json.loads(SC_JSON.read_text())["referee"]["deembedded"]["total"] \
        if SC_JSON.exists() else None
    return {
        "what": ("area-exact Greenhouse + PEC ground image, L(strip) - L(bridge); the bridge is "
                 "the physical short standard's: M2 arm outer plane -> post centre, M1 arm "
                 "post centre -> inner plane"),
        "total": total, "strip": float(strip.total), "bridge": float(bridge.total),
        "strip_worst_error": float(strip.worst_error),
        "bridge_worst_error": float(bridge.worst_error),
        "ground_height": GROUND_H, "dz_underpass": dz0,
        "reference_planes_x": [xo, xi], "post_x": [x0, x1], "split_on_m2": split_c,
        "post_split_band": band,
        "post_split_band_rel": (max(vals) - min(vals)) / total,
        "underpass_depth": depth, "via": via,
        "lead_columns": "z-directed in the DUT and the short alike: they cancel term by term",
        "old_study_deembedded": dee_old,
        "same_as_old_study": (abs(total / dee_old - 1.0) if dee_old else None),
    }


# ----------------------------------------------------------------------------
# 4. the harvested GPU lanes

def latest_run(lane: str, fname: str) -> tuple[pathlib.Path | None, dict[str, Any]]:
    """The newest harvested ``fdfd-gpu-<lane>-<utc>/<fname>`` (``{}`` if none)."""
    cands = sorted(p for p in RUNS.glob(f"fdfd-gpu-{lane}-*") if (p / fname).exists())
    if not cands:
        return None, {}
    return cands[-1], json.loads((cands[-1] / fname).read_text())


def all_runs(lane: str, fname: str) -> list[tuple[pathlib.Path, dict[str, Any]]]:
    """Every harvested ``fdfd-gpu-<lane>-<utc>/<fname>``, oldest first.

    Used where a REPLICATE matters: a gate whose number is a difference
    between two solver runs has to be read against how much that number
    itself moves from job to job."""
    return [(d, json.loads((d / fname).read_text()))
            for d in sorted(RUNS.glob(f"fdfd-gpu-{lane}-*")) if (d / fname).exists()]


def richardson(h: Sequence[float], values: Sequence[float]) -> dict[str, Any]:
    """``spiral_convergence.richardson`` (p = 1 and p = 2 from the finest
    pair, the observed order from the finest three), reused unmodified."""
    return load_sc().richardson(list(h), list(values))


def family(levels: dict[str, Any]) -> dict[str, Any]:
    """One level family as sorted arrays + its Richardson."""
    ms = sorted((int(v["m"]), v) for v in levels.values() if v.get("L_dut") is not None)
    out: dict[str, Any] = {
        "m": [m for m, _ in ms], "h": [WIDTH / m for m, _ in ms],
        "n_unknowns": [v["grid"]["n_unknowns"] for _, v in ms],
        "L_dut": [v["L_dut"] for _, v in ms],
        "dL_dwidth": [(v.get("grad") or [None] * 3)[2] for _, v in ms]}
    if len(ms) >= 2:
        out["richardson"] = richardson(out["h"], out["L_dut"])
    return out


def p6_block(p0: dict[str, Any]) -> dict[str, Any]:
    """Gate P6 from the probe: each level-dependence of the OLD fixture
    switched to its physical definition alone, at W/1 and W/2."""
    src = (p0.get("p6") or {}).get("levels") or {}
    out: dict[str, Any] = {"wall_w": (p0.get("p6") or {}).get("wall_w"), "levels": {}}
    for lvl, cases in src.items():
        base = cases.get("old", {}).get("L_dut")
        rec: dict[str, Any] = {}
        for name, r in cases.items():
            if r.get("L_dut") is None or base is None:
                continue
            rec[name] = {"L_dut": r["L_dut"], "rel_to_old": r["L_dut"] / base - 1.0,
                         # the same pair read the other way: how far the OLD
                         # fixture's L sits from the variant's (L_old / L_case - 1)
                         "old_rel_to_case": base / r["L_dut"] - 1.0,
                         "n_unknowns": r["grid"]["n_unknowns"], "lid_z": r["grid"]["lid_z"],
                         "wall_x": r["grid"]["wall_x"][1]}
        if all(k in rec for k in ("a_walls", "b_short", "c_vertical", "abc")):
            s = sum(rec[k]["rel_to_old"] for k in ("a_walls", "b_short", "c_vertical"))
            rec["sum_of_individual"] = s
            # (a) + (b): the two level-dependences that move a PHYSICAL object
            # ((c) moves none -- ``p6_objects``)
            rec["objects_ab"] = rec["a_walls"]["rel_to_old"] + rec["b_short"]["rel_to_old"]
            rec["interaction"] = rec["abc"]["rel_to_old"] - s
            if "new" in rec:
                rec["new_minus_abc"] = rec["new"]["L_dut"] / rec["abc"]["L_dut"] - 1.0
        out["levels"][lvl] = rec
    return out


P6_OBJECT_DIVS = (2, 3)


def p6_objects(divs: Sequence[int] = P6_OBJECT_DIVS) -> dict[str, Any]:
    """Which of P6's three old level-dependences changes a PHYSICAL object:
    the old fixture at W/div against the same fixture with (a), (b) or (c)
    switched, read back from the BUILT cell masks and port boxes
    (``fixture_coordinates``, as gate P1). (a) moves the walls and the lid,
    (b) the post and the bridge arms; (c) -- the level-1 z lines subdivided,
    the port gap held at 10 um, which is also the old fixture's port gap
    (one frozen 10 um cell) -- moves no object at all: it only adds z lines,
    i.e. (c) is a discretisation left un-refined, not a fixture change."""
    out: dict[str, Any] = {"divs": list(divs), "tolerance_m": 1e-12, "per_div": {}}
    for div in divs:
        base = build(spec_old(div))
        cb = fixture_coordinates(base)
        fb = _flatten({k: v for k, v in cb.items() if k != "volume"})
        rec: dict[str, Any] = {"old_z_lines": int(len(base.z))}
        for name, kw in (("a_walls", {"walls": True}), ("b_short", {"short": True}),
                         ("c_vertical", {"vertical": True})):
            mdl = build(spec_old(div, **kw))
            cm = fixture_coordinates(mdl)
            fm = _flatten({k: v for k, v in cm.items() if k != "volume"})
            devs = {k: abs(fm.get(k, np.inf) - v) for k, v in fb.items()
                    if np.isfinite(v) or np.isfinite(fm.get(k, np.nan))}
            devs = {k: (0.0 if (np.isnan(d) and not np.isfinite(fb[k])) else d)
                    for k, d in devs.items()}
            worst_k = max(devs, key=lambda k: devs[k]) if devs else ""
            vol = max(abs(cm["volume"][k] / v - 1.0) for k, v in cb["volume"].items())
            moved = sorted({k.split(".")[0] for k, d in devs.items() if d > out["tolerance_m"]})
            rec[name] = {"worst_coordinate_deviation_m": float(devs.get(worst_k, 0.0)),
                         "worst_coordinate_key": worst_k,
                         "worst_volume_rel_deviation": float(vol),
                         "objects_moved": moved, "z_lines": int(len(mdl.z)),
                         "changes_a_physical_object": bool(
                             devs.get(worst_k, 0.0) > out["tolerance_m"] or vol > 1e-12)}
            del mdl
        out["per_div"][str(div)] = rec
    return out


def plateau_block(p0: dict[str, Any]) -> dict[str, Any]:
    pl = p0.get("plateau") or {}
    runs = pl.get("runs") or []
    main = [r for r in runs if r["freq"] == FREQ]
    out: dict[str, Any] = {"level": pl.get("level"), "wall_w": pl.get("wall_w"),
                           "grid": pl.get("grid"), "metal_cells": (p0.get("fixture") or {}).get("metal_cells"),
                           "sigma": [r["sigma"] for r in main], "L_dut": [r["L_dut"] for r in main],
                           "skin_over_width": [r["skin_over_width"] for r in main],
                           "re_z_diff": [r["re_z_diff"] for r in main]}
    pick = {r["sigma"]: r for r in main}
    if SIGMA in pick:
        ref = pick[SIGMA]["L_dut"]
        out["rel_to_chosen"] = [v / ref - 1.0 for v in out["L_dut"]]
        twin = [r for r in runs if r["freq"] == FREQ_RC and r["sigma"] == SIGMA_RC]
        if twin:
            out["rc_twin"] = {"freq": FREQ_RC, "sigma": SIGMA_RC, "L_dut": twin[0]["L_dut"],
                              "re_z_diff": twin[0]["re_z_diff"],
                              "skin_depth": twin[0]["skin_depth"],
                              "rel_to_chosen": twin[0]["L_dut"] / ref - 1.0}
    return out


# ----------------------------------------------------------------------------
# 5. gates

V1_TOL = 0.05              # the V1 tolerance of the old study, not widened
V1_TIGHT = 0.03            # reported beside it
AD_FD_TOL = 1e-4           # rule 1 of the task
# H1: hybrid (host-backed) memory must not change the answer. Both runs are
# the SAME cuDSS factorisation algorithm on the same card, differing only in
# where the factor is held, so the bound is the solver's own reproducibility
# (cuDSS's triangular solve is not bit-reproducible across calls: 4.2e-16
# relative, selftest in rfx/fdfd/_cudss.py), not a physics tolerance
HYBRID_TOL = 1e-9
H2_LEVEL = 5               # the level the hybrid option is FOR
# P6's identity check: at level 1 the physical short standard and the jointly
# refined z grid ARE the old fixture's (post [22, 32] um, z unchanged), so L
# must agree to solver noise; the bound is 10 x the cuDSS-vs-SuperLU L_dut
# agreement J0 measured on the W/3 fixture (9.2e-10, run 369367261104)
P6_IDENTITY_TOL = 1e-8


def evaluate_gates(study: dict[str, Any]) -> dict[str, Any]:
    g: dict[str, Any] = {}
    ref = study["referee"]["total"]
    p1 = study.get("p1") or {}
    g["P1"] = {"what": "fixture invariance from the built masks: every wall / lid / post / "
                       "bridge / column / port / interface coordinate identical across the "
                       "levels and every level-1 line a line of every finer level",
               "levels": p1.get("levels"),
               "worst_coordinate_deviation_m": p1.get("worst_coordinate_deviation_m"),
               "nesting_worst_distance_m": p1.get("nesting_worst_distance_m"),
               "worst_volume_rel_deviation": p1.get("worst_volume_rel_deviation"),
               "tolerance_m": p1.get("tolerance_m"),
               "old_fixture_worst_deviation_m": (study.get("p1_old_fixture") or {}).get(
                   "worst_coordinate_deviation_m"),
               "passed": bool(p1.get("passed"))}
    fam = study.get("family") or {}
    r = fam.get("richardson") or {}
    L = fam.get("L_dut") or []
    ms = fam.get("m") or []
    d = [b - a for a, b in zip(L[:-1], L[1:])]
    last3 = d[-2:] if len(d) >= 2 else []
    mono3 = bool(len(last3) == 2 and last3[0] * last3[1] > 0)
    mono_all = bool(d and all(x * d[0] > 0 for x in d))
    g["P2"] = {"what": "monotone convergence over the finest three levels with an observed "
                       "order, and the Richardson range",
               "levels": ms, "L_dut": L, "differences": d,
               "monotone_finest_three": mono3, "monotone_all": mono_all,
               "contracting": bool(len(last3) == 2 and abs(last3[1]) < abs(last3[0])),
               "observed_order": r.get("observed_order"),
               "estimates": r.get("estimates"), "range": r.get("range"),
               "passed": bool(mono3 and r.get("observed_order") is not None)}
    if r.get("range"):
        rel = [v / ref - 1.0 for v in r["range"]]
        worst = max(abs(v) for v in rel)
        g["P3"] = {"what": "the V1 answer: the extrapolated range against the referee "
                           f"{ref * 1e12:.3f} pH (one number for every level)",
                   "referee": ref, "L_range": r["range"], "rel_range": rel,
                   "per_level_gap": {f"m={m}": v / ref - 1.0 for m, v in zip(ms, L)},
                   "tolerance": V1_TOL, "within_5_percent": bool(worst <= V1_TOL),
                   "within_3_percent": bool(worst <= V1_TIGHT),
                   "passed": bool(worst <= V1_TOL)}
    else:
        g["P3"] = {"passed": False, "why": "no Richardson range (levels missing)"}
    fd = study.get("fd_width") or {}
    g["P4"] = {"what": "jax.grad of L_dut vs FD4 (1 % step) of dL/dwidth through cuDSS at the "
                       "finest level that has it",
               "level": fd.get("level"), "ad": fd.get("ad"), "fd4": fd.get("fd"),
               "rel": fd.get("rel"), "step": fd.get("step"), "tolerance": AD_FD_TOL,
               "passed": bool(fd.get("rel") is not None and fd["rel"] <= AD_FD_TOL)}
    gw = [v for v in (fam.get("dL_dwidth") or []) if v is not None]
    ch = [abs(b - a) for a, b in zip(gw[:-1], gw[1:])]
    g["P5"] = {"what": "dL/dwidth level-to-level changes decreasing",
               "levels": ms, "dL_dwidth": gw, "abs_changes": ch,
               "passed": bool(len(ch) >= 2 and all(b < a for a, b in zip(ch[:-1], ch[1:])))}
    hb = study.get("hybrid") or {}
    h1 = hb.get("h1") or {}
    g["H1"] = {"what": "cuDSS hybrid (host-backed) memory with an EXPLICIT device limit BELOW "
                       "the factor size returns the in-memory run's L_dut and gradient (the "
                       "gradient as a VECTOR, the metric tests/unit/fdfd/test_fdfd_linear_solve.py "
                       "uses for two backends) in the same job on the same card, with the limit "
                       "binding THROUGH THE SHIPPED PATH: the plan record says the limit was "
                       "applied at DirectSolver construction and nowhere else "
                       "(plan_limit_applied = \"constructor\"), cuDSS's plan then asks for the "
                       "limit instead of the factor size, and the factor's bytes appear in the "
                       "process's resident set. The controls -- the SAME in-memory computation "
                       "run twice, the same level on the other card, and the spread of this "
                       "comparison across jobs (h1_replicates) -- are the floor the two "
                       "differences are read against",
               "level": h1.get("level"), "n_unknowns": h1.get("n_unknowns"),
               "device_memory_limit_gb": h1.get("device_memory_limit_gb"),
               "in_core_factor_gb": h1.get("in_core_factor_gb"),
               "limit_below_in_core_factor": h1.get("limit_below_in_core_factor"),
               "rel_L": h1.get("rel_L"), "rel_grad_l2": h1.get("rel_grad_l2"),
               "rel_grad_worst": h1.get("rel_grad_worst"),
               # the control: the SAME in-memory computation run twice in the
               # same job -- cuDSS is not bit-reproducible across calls, so this
               # is the floor any hybrid-vs-in-core difference is read against
               "control_in_core_repeat_rel_L": (h1.get("control_in_core_repeat")
                                                or {}).get("rel_L"),
               "control_in_core_repeat_rel_grad_l2": (h1.get("control_in_core_repeat")
                                                      or {}).get("rel_grad_l2"),
               "tolerance": HYBRID_TOL,
               "slowdown_value_and_grad": h1.get("slowdown_value_and_grad"),
               # and the limit BOUND: cuDSS's plan for the same matrix came back
               # at the limit instead of the factor size, and the factor's bytes
               # appeared in the PROCESS's resident set instead
               "plan_permanent_device_gb": h1.get("plan_permanent_device_gb"),
               "plan_permanent_host_gb": h1.get("plan_permanent_host_gb"),
               # WHERE the limit was applied when those plans were taken:
               # "constructor" is the shipped path (rfx/fdfd/_cudss.py,
               # _execution_options) and the only one this gate accepts
               "plan_limit_applied": h1.get("plan_limit_applied"),
               "plan_rel_deviation_from_limit": h1.get("plan_rel_deviation_from_limit"),
               "limit_binds_the_plan": h1.get("limit_binds_the_plan"),
               # the SOLUTION VECTOR of the one-fixture probe, and its own
               # in-core-vs-in-core control: L_dut is a functional of x, and it
               # agrees far closer than x does
               "solution_rel_diff": h1.get("solution_rel_diff"),
               "solution_rel_diff_control": h1.get("solution_rel_diff_control"),
               "host_rss_growth_hybrid_gb": h1.get("host_rss_growth_hybrid_gb"),
               "rel_grad_l2_over_control": h1.get("rel_grad_l2_over_control"),
               # and how much the reported difference itself moves between jobs
               "rel_grad_l2_across_jobs": ((hb.get("h1_replicates") or {}).get(
                   "rel_grad_l2_range")),
               "L_within_tolerance": bool(h1.get("rel_L") is not None
                                          and h1["rel_L"] <= HYBRID_TOL),
               "grad_within_tolerance": bool(h1.get("rel_grad_l2") is not None
                                             and h1["rel_grad_l2"] <= HYBRID_TOL),
               "passed": bool(h1.get("rel_L") is not None
                              and h1.get("limit_below_in_core_factor")
                              and h1.get("limit_binds_the_plan")
                              and (h1.get("host_rss_growth_hybrid_gb") or 0.0) > 0.0
                              and h1["rel_L"] <= HYBRID_TOL
                              and (h1.get("rel_grad_l2") or 1.0) <= HYBRID_TOL)}
    m5 = hb.get("m5") or {}
    has5 = H2_LEVEL in ms
    if has5:
        i = ms.index(H2_LEVEL)
        prev = d[i - 2:i]          # the two steps ending at m = 5
        g["H2"] = {"what": f"m = {H2_LEVEL} continues the monotone sequence (same sign, "
                           "smaller step) and is the ladder's finest level",
                   "levels": ms, "L_dut": L, "differences": d,
                   "step_into_level": d[i - 1] if i >= 1 else None,
                   "monotone": bool(len(prev) == 2 and prev[0] * prev[1] > 0),
                   "contracting": bool(len(prev) == 2 and abs(prev[1]) < abs(prev[0])),
                   "passed": bool(len(prev) == 2 and prev[0] * prev[1] > 0
                                  and abs(prev[1]) < abs(prev[0]))}
    else:
        g["H2"] = {"what": f"m = {H2_LEVEL} continues the monotone sequence",
                   "levels": ms, "measured": False,
                   "why": m5.get("not_run") or m5.get("error")
                   or "no m = 5 record in any harvested lane",
                   "in_core_factor_gb": m5.get("in_core_factor_gb"),
                   "host_needed_gb": m5.get("host_needed_gb"),
                   "host_allowed_gb": m5.get("host_allowed_gb"),
                   "cgroup_limit_gb": m5.get("cgroup_limit_gb"),
                   "node_mem_total_gb": m5.get("node_mem_total_gb"),
                   "passed": False}
    p6 = study.get("p6") or {}
    lv1 = (p6.get("levels") or {}).get("1") or {}
    ident = [abs(lv1[k]["rel_to_old"]) for k in ("b_short", "c_vertical") if k in lv1]
    objs = ((p6.get("objects") or {}).get("per_div") or {}).get("2") or {}
    g["P6"] = {"what": "old vs new: each level-dependence of the old fixture switched alone, "
                       "at levels 1 and 2 (relative to the old fixture at the same level): "
                       "(a) walls and (b) short standard move a physical object, (c) the "
                       "vertical grid refined moves none (a discretisation left un-refined)",
               "changes_a_physical_object_at_level_2": {
                   k: v.get("changes_a_physical_object") for k, v in objs.items()
                   if isinstance(v, dict)},
               "per_level": {k: {kk: (vv["rel_to_old"] if isinstance(vv, dict) else vv)
                                 for kk, vv in v.items()} for k, v in (p6.get("levels") or {}).items()},
               "level1_identity_worst": max(ident) if ident else None,
               "identity_tolerance": P6_IDENTITY_TOL,
               "passed": bool(len(ident) == 2 and max(ident) <= P6_IDENTITY_TOL
                              and "2" in (p6.get("levels") or {}))}
    return g


# ----------------------------------------------------------------------------
# 6. assembly

P1_LEVELS = (1, 2, 3, 4, 6)          # built on the host for gate P1 (6: built, never solved)
LEDGER = RUNS / "p_run_ledger.json"   # every VESSL run of this study (id, state, wall time)


def p1_blocks(levels: Sequence[int] = P1_LEVELS) -> tuple[dict[str, Any], dict[str, Any]]:
    """Gate P1 on the primary family and the same read-back on the OLD
    fixture at W/1..W/3 (the contrast: there it fails by construction)."""
    models = {m: build(spec_new(m, WALL_W)) for m in levels}
    p1 = p1_gate(models)
    p1["wall_w"] = WALL_W
    p1["metal_cells"] = METAL_CELLS
    p1["n_unknowns"] = {str(m): int(models[m].n_unknowns) for m in levels}
    del models
    old = {m: build(spec_old(m)) for m in (1, 2, 3)}
    po = p1_gate(old)
    po = {k: po[k] for k in ("levels", "worst_coordinate_deviation_m", "worst_coordinate_key",
                             "worst_volume_rel_deviation", "nesting_worst_distance_m", "passed")}
    return p1, po


def assemble(reuse: dict[str, Any] | None = None) -> dict[str, Any]:
    reuse = reuse or {}
    t0 = time.time()
    study: dict[str, Any] = {
        "what": "study P: the level-invariant spiral fixture and its joint GPU convergence ladder",
        "fixture": {"freq": FREQ, "n_turns": N_TURNS, "r_out": R_OUT, "spacing": SPACING,
                    "width": WIDTH, "lead": LEAD, "margin": MARGIN, "z_top": Z_TOP,
                    "base_dz_level1": BASE_DZ, "metal_cells_level1": METAL_CELLS,
                    "pad_ratio": PAD_RATIO, "short_gap": SHORT_GAP, "port_gap": PORT_GAP,
                    "wall_margin_w": WALL_W, "sigma": SIGMA,
                    "skin_depth": skin_depth(FREQ, SIGMA),
                    "skin_over_width": skin_depth(FREQ, SIGMA) / WIDTH,
                    "skin_over_thickness": skin_depth(FREQ, SIGMA) / T_M2,
                    "dielectrics": "vacuum (eps_si = eps_ox = 1)",
                    "metal": "uniform-current volumetric (solve_spiral(sigma_volumetric=))"},
        "runs": json.loads(LEDGER.read_text()) if LEDGER.exists() else [],
    }
    study["referee"] = reuse.get("referee") or referee_block()
    if reuse.get("p1") and reuse.get("p1_old_fixture"):
        study["p1"], study["p1_old_fixture"] = reuse["p1"], reuse["p1_old_fixture"]
    else:
        study["p1"], study["p1_old_fixture"] = p1_blocks()
    d0, p0 = latest_run("p0", "p0_probe.json")
    study["probe_source"] = str(d0.relative_to(REPO)) if d0 else None
    if p0:
        study["plans"] = {k: {kk: v.get(kk) for kk in (
            "wall_w", "metal_cells", "m", "n", "permanent_device_memory_gb",
            "peak_device_memory_gb", "hybrid_min_device_memory_gb", "plan_seconds",
            "assembly_seconds", "error")} for k, v in {**(p0.get("plans") or {}),
                                                      **(p0.get("bigplans") or {})}.items()}
        w = p0.get("walls") or {}
        study["walls"] = {k: w.get(k) for k in ("level", "margins_w", "L_dut",
                                                  "rel_change_to_next", "chosen_margin_w", "rule")}
        study["walls"]["metal_cells"] = (p0.get("fixture") or {}).get("metal_cells")
        study["walls"]["n_unknowns"] = [w["runs"][f"{m:g}"]["grid"]["n_unknowns"]
                                        for m in w.get("margins_w", [])]
        study["p6"] = p6_block(p0)
        study["p6"]["objects"] = (reuse.get("p6") or {}).get("objects") or p6_objects()
        study["sigma_plateau"] = plateau_block(p0)
    fams: dict[str, dict[str, Any]] = {"primary": {}, "alt": {}}
    fd_width: dict[str, Any] = {}
    thru: dict[str, Any] = {}
    factor: dict[str, Any] = {}
    srcs = []
    runs_ = [(dd, json.loads((dd / fname).read_text()))
             for lane, fname in (("p1", "p1_ladder.json"), ("p1b", "p1b_ladder.json"),
                                 ("p1c", "p1c_ladder.json"), ("p3", "p3_ladder.json"))
             for dd in sorted(RUNS.glob(f"fdfd-gpu-{lane}-*")) if (dd / fname).exists()]
    # oldest first: a level measured again by a later run replaces the earlier one
    for dd, rr in sorted(runs_, key=lambda t: t[0].name.rsplit("-", 1)[-1]):
        lane = dd.name.split("-")[2]
        cache = (rr.get("config") or {}).get("factor_cache_size")
        srcs.append(str(dd.relative_to(REPO)))
        for k, v in (rr.get("levels") or {}).items():
            if v.get("L_dut") is not None:
                fams["primary"][k] = dict(v, source=lane, factor_cache_size=cache)
                if v.get("fd_width"):
                    fd_width = dict(v["fd_width"], level=int(k))
        for k, v in (rr.get("xlevels") or {}).items():
            if v.get("L_dut") is not None:
                fams["alt"][str(v["m"])] = dict(v, source=lane)
        for k, v in ((rr.get("thru") or {}).get("levels") or {}).items():
            thru[k] = v
        thru_len = (rr.get("thru") or {}).get("bar_length")
        if thru_len:
            thru["_bar_length"] = thru_len
    if p0:
        for k, v in ((p0.get("ladder") or {}).get("levels") or {}).items():
            if v.get("L_dut") is not None and k not in fams["alt"]:
                fams["alt"][k] = dict(v, m=int(k), source="p0")
    study["ladder_sources"] = srcs
    for k, v in fams["primary"].items():
        f = v.get("factor", {}).get("per_backend", {}).get("cudss", {})
        st = (f.get("stats") or {}).get("A", {})
        # with a factor cache of 0 the probe's repeat solve factorises again, so
        # the factorisation time is the repeat's wall time (factor + 2 solves);
        # with a cache it is cold minus warm
        fs = f.get("warm_seconds") if v.get("factor_cache_size") == 0 else f.get("factor_seconds")
        factor[k] = {"n_unknowns": v["grid"]["n_unknowns"], "shape": v["grid"]["shape"],
                     "factor_seconds": fs, "factor_cache_size": v.get("factor_cache_size"),
                     "solve_seconds": (f.get("solve_seconds") if v.get("factor_cache_size")
                                       else None),
                     "residual": f.get("residual"),
                     "plan_permanent_device_gb": ((study.get("plans") or {}).get(
                         f"W{WALL_W:g}_mc{v.get('metal_cells', METAL_CELLS)}_m{k}") or {}).get(
                         "permanent_device_memory_gb"),
                     "lu_nnz": st.get("lu_nnz"),
                     "factor_permanent_device_gb": st.get("factorization_permanent_device_gb"),
                     "device_gb_after_factor": f.get("device_memory_gb_after", {}).get("used"),
                     "value_and_grad_seconds": v.get("value_and_grad_seconds"),
                     "device_gb_after_value_and_grad": (v.get("device_memory_gb_peak_after")
                                                        or {}).get("used"),
                     "source": v["source"]}
    study["levels"] = {k: {kk: v.get(kk) for kk in ("m", "metal_cells", "L_dut", "grad", "grid",
                                                   "value_and_grad_seconds", "source")}
                       for k, v in sorted(fams["primary"].items(), key=lambda t: int(t[0]))}
    study["cost"] = factor
    study["family"] = family(fams["primary"])
    twin = {k: {"m": v["m"], "grid": v["grid"], "L_dut": v["rc_twin"]["L_dut"]}
            for k, v in fams["primary"].items() if (v.get("rc_twin") or {}).get("L_dut")}
    study["family_rc_twin"] = family(twin)
    study["family_rc_twin"]["protocol"] = {"freq": FREQ_RC, "sigma": SIGMA_RC,
                                           "skin_depth": skin_depth(FREQ_RC, SIGMA_RC)}
    if twin:
        study["family_rc_twin"]["rel_to_primary"] = [
            a / b - 1.0 for a, b in zip(study["family_rc_twin"]["L_dut"],
                                        [fams["primary"][str(m)]["L_dut"]
                                         for m in study["family_rc_twin"]["m"]])]
    study["family_alt"] = family(fams["alt"])
    study["family_alt"]["metal_cells_level1"] = METAL_CELLS_ALT
    study["fd_width"] = fd_width
    if thru:
        bl = thru.pop("_bar_length", None)
        study["thru"] = {"bar_length": bl, "levels": {
            k: {"L_deembedded": v.get("L_dut"), "L_raw": v.get("L_raw"),
                "n_unknowns": (v.get("grid") or {}).get("n_unknowns"),
                "over_referee": (v["L_dut"] / study["referee"]["total"]) if v.get("L_dut") else None}
            for k, v in sorted(thru.items(), key=lambda t: int(t[0]))}}
    study["hybrid"] = hybrid_block(study)
    study["paper"] = paper_block()
    study["corrections"] = corrections(study)
    study["extrapolation_checks"] = extrapolation_checks(study)
    study["residual"] = residual_block(study)
    study["resolution_for_3_percent"] = resolution_needed(study)
    study["gates"] = evaluate_gates(study)
    study["seconds_assembly"] = time.time() - t0
    return study


def paper_referee() -> dict[str, Any]:
    """Study H's own referee for the paper geometry (``rfic_spiral.py``'s
    ``referee_deembedded``: area-exact strip minus the short's bridge split at
    the post centre, ground image at its 131.54 um), imported unmodified."""
    rf = _load(RFIC_PATH, "rfic_spiral_for_p")
    sg, cv = rf.load_referee(), rf.bind_referee()
    st, br = rf.strip_terms(sg, cv, rf.THETA0), rf.bridge_terms(sg, cv, rf.THETA0)
    return {"total": rf.referee_deembedded(sg, cv, rf.THETA0), "strip": float(st.total),
            "bridge": float(br.total), "ground_height": rf.GROUND_H,
            "note": "rfic_spiral.referee_deembedded at its THETA0; the physical post "
                    "(30 um from each column) is centred on the split-0.5 transition"}


def paper_block() -> dict[str, Any]:
    """The secondary geometry from the P2 lane (no gate: reported)."""
    d, r = latest_run("p2", "p2_paper.json")
    if not r:
        return {}
    ref = paper_referee()
    out: dict[str, Any] = {"source": str(d.relative_to(REPO)), "fixture": r.get("fixture"),
                           "referee": ref}
    out["plans"] = {k: {kk: v.get(kk) for kk in ("n", "permanent_device_memory_gb",
                                                 "hybrid_min_device_memory_gb", "error")}
                    for k, v in (r.get("plans") or {}).items()}
    w = (r.get("walls") or {}).get("runs") or {}
    out["walls_level1"] = {k: v.get("L_dut") for k, v in w.items()}
    pl = r.get("plateau") or {}
    runs = pl.get("runs") or []
    main = [x for x in runs if x["freq"] == PAPER_FREQ]
    out["plateau"] = {"level": pl.get("level"), "grid": pl.get("grid"),
                      "sigma": [x["sigma"] for x in main], "L_dut": [x["L_dut"] for x in main],
                      "skin_over_width": [x["skin_over_width"] for x in main],
                      "error": pl.get("error")}
    pick = {x["sigma"]: x["L_dut"] for x in main}
    if PAPER_SIGMA in pick:
        out["plateau"]["rel_to_chosen"] = [v / pick[PAPER_SIGMA] - 1.0 for v in out["plateau"]["L_dut"]]
        tw = [x for x in runs if x["freq"] == PAPER_FREQ_RC]
        if tw:
            out["plateau"]["rc_twin"] = {"freq": PAPER_FREQ_RC, "sigma": PAPER_SIGMA_RC,
                                         "L_dut": tw[0]["L_dut"],
                                         "rel_to_chosen": tw[0]["L_dut"] / pick[PAPER_SIGMA] - 1.0}
    lv = {k: v for k, v in (r.get("levels") or {}).items() if v.get("L_dut") is not None}
    out["levels"] = {k: {"L_dut": v["L_dut"], "n_unknowns": v["grid"]["n_unknowns"],
                         "gap": v["L_dut"] / ref["total"] - 1.0, "seconds": v.get("seconds"),
                         "device_gb_after": v.get("device_memory_gb_after")}
                     for k, v in sorted(lv.items(), key=lambda t: int(t[0]))}
    out["errors"] = {k: v.get("error") for k, v in (r.get("levels") or {}).items() if v.get("error")}
    if len(lv) >= 2:
        ks = sorted(lv, key=int)
        rr = richardson([PAPER["width"] / int(k) for k in ks], [lv[k]["L_dut"] for k in ks])
        out["richardson"] = rr
        out["rel_range"] = [v / ref["total"] - 1.0 for v in rr["range"]]
    return out


def hybrid_block(study: dict[str, Any] | None = None) -> dict[str, Any]:
    """The P3 lane: cuDSS hybrid (host + device) memory with an explicit
    device limit, and what a level-5 factor would need in HOST memory.

    Gate H1's raw numbers (level 3 in hybrid mode with a device limit below
    the factor size, against the in-memory run of the same level in the same
    job) and the level-5 account: cuDSS's plan estimates, the container's
    cgroup memory limit, and whether the solve was attempted. With ``study``
    given, H1's in-memory run is also compared with the LADDER's own record
    of that level -- a different card, a different job, the same protocol.

    Three things here are about reading H1's difference rather than taking
    it: ``plan_limit_applied`` (the PROVENANCE of the plan numbers -- only a
    plan whose limit was applied at ``DirectSolver`` construction, the
    shipped path, is accepted as evidence that the limit binds),
    ``solution_rel_diff``/``_control`` (the probe's solution VECTOR against
    the same solve run again in-core) and ``h1_replicates`` (the same
    comparison in every P3 job)."""
    d, r = latest_run("p3", "p3_ladder.json")
    if not r:
        return {}
    out: dict[str, Any] = {
        "source": str(d.relative_to(REPO)),
        "option": "RFX_FDFD_CUDSS_HYBRID=1 with RFX_FDFD_CUDSS_HYBRID_LIMIT "
                  "(rfx/fdfd/_cudss.py, \"Device memory\"): cuDSS keeps the factor in host "
                  "memory and streams it, and the limit bounds ALL of its device memory. "
                  "Without a limit cuDSS assumes the whole card, which is how the first "
                  "m = 5 attempt (run 369367261252) reached ALLOC_FAILED.",
        "host": r.get("host")}
    h1 = r.get("h1") or {}
    if h1.get("gate"):
        pl = h1.get("plans") or {}
        pr = h1.get("probe") or {}
        vg = h1.get("value_and_grad") or {}
        out["h1"] = {
            "level": h1.get("level"), "n_unknowns": (h1.get("grid") or {}).get("n_unknowns"),
            "device_memory_limit": h1.get("limit"), "device_memory_limit_gb": h1.get("limit_gb"),
            "in_core_factor_gb": (pl.get("in_core") or {}).get("permanent_device_memory_gb"),
            "hybrid_min_device_memory_gb": (pl.get("hybrid") or {}).get(
                "hybrid_min_device_memory_gb"),
            "limit_below_in_core_factor": h1.get("limit_below_in_core_factor"),
            "limit_at_least_hybrid_min": h1.get("limit_at_least_hybrid_min"),
            "factor_seconds": {k: (pr.get(k) or {}).get("factor_seconds")
                               for k in ("in_core", "hybrid")},
            "solve_seconds": {k: (pr.get(k) or {}).get("solve_seconds")
                              for k in ("in_core", "hybrid")},
            "residual": {k: (pr.get(k) or {}).get("residual") for k in ("in_core", "hybrid")},
            "device_peak_used_gb": {k: (pr.get(k) or {}).get("device_peak_used_gb")
                                    for k in ("in_core", "hybrid")},
            "plan_permanent_device_gb": {k: (pl.get(k) or {}).get("permanent_device_memory_gb")
                                         for k in ("in_core", "hybrid")},
            "plan_permanent_host_gb": {k: (pl.get(k) or {}).get("permanent_host_memory_gb")
                                       for k in ("in_core", "hybrid")},
            # ru_maxrss is a HIGH-WATER mark, so the in-core run's is the
            # in-core figure and the hybrid run's is after the factor moved to
            # the host: the growth is what hybrid mode put there
            "host_peak_rss_gb": {k: ((vg.get(k) or {}).get("host_memory_gb") or {}).get("peak_rss")
                                 for k in ("in_core", "hybrid", "in_core_repeat")},
            "host_cgroup_current_gb": {
                k: ((vg.get(k) or {}).get("cgroup") or {}).get("memory.current_gb")
                for k in ("in_core", "hybrid", "in_core_repeat")},
            "solution_rel_diff": pr.get("solution_rel_diff"),
            # the same solve run again IN-CORE: the floor for the line above
            # (cuDSS's triangular solve is not bit-reproducible across calls)
            "solution_rel_diff_control": pr.get("solution_rel_diff_control"),
            "device_baseline_used_gb": {k: (pr.get(k) or {}).get("device_baseline_used_gb")
                                        for k in ("in_core", "hybrid")},
            "factor_slowdown": pr.get("factor_slowdown"),
            "solve_slowdown": pr.get("solve_slowdown"),
            "value_and_grad_seconds": {k: (vg.get(k) or {}).get("value_and_grad_seconds")
                                       for k in ("in_core", "hybrid")},
            "value_and_grad_device_peak_used_gb": {
                k: (vg.get(k) or {}).get("device_peak_used_gb") for k in ("in_core", "hybrid")},
            # PROVENANCE of the plan numbers below: "constructor" means the
            # limit reached cuDSS at DirectSolver construction and nowhere
            # else (rfx/fdfd/_cudss.py, hybrid_record). A plan record without
            # it was taken by a code path that is not the shipped one, and
            # limit_binds_the_plan refuses to read it as evidence
            "plan_limit_applied": (pl.get("hybrid") or {}).get("hybrid_limit_applied"),
            **{k: h1["gate"].get(k) for k in ("L_in_core", "L_hybrid", "rel_L", "grad_in_core",
                                              "grad_hybrid", "rel_grad", "rel_grad_worst",
                                              "rel_grad_l2", "tolerance",
                                              "slowdown_value_and_grad", "passed")},
            "control_in_core_repeat": h1["gate"].get("control_in_core_repeat")}
        hh = out["h1"]
        lim, plan_dev = hh["device_memory_limit_gb"], (hh["plan_permanent_device_gb"] or {})
        # does the limit BIND? cuDSS's own plan for the same matrix, in hybrid
        # mode, must come back at the limit instead of the in-core factor size
        hh["limit_binds_the_plan"] = bool(
            lim and plan_dev.get("hybrid") is not None
            and hh.get("plan_limit_applied") == "constructor"
            and abs(plan_dev["hybrid"] - lim) <= 1e-6 * lim)
        hh["plan_rel_deviation_from_limit"] = (
            abs(plan_dev["hybrid"] - lim) / lim
            if lim and plan_dev.get("hybrid") is not None else None)
        rss = hh["host_peak_rss_gb"]
        hh["host_rss_growth_hybrid_gb"] = (
            (rss.get("hybrid") - rss.get("in_core"))
            if rss.get("hybrid") is not None and rss.get("in_core") is not None else None)
        ctl = (hh.get("control_in_core_repeat") or {}).get("rel_grad_l2")
        hh["rel_grad_l2_over_control"] = (hh["rel_grad_l2"] / ctl
                                          if ctl and hh.get("rel_grad_l2") else None)
        lv = ((study or {}).get("levels") or {}).get(str(hh["level"])) or {}
        if lv.get("L_dut") and lv.get("grad") and hh.get("L_in_core"):
            gl, gh_ = np.asarray(lv["grad"]), np.asarray(hh["grad_in_core"])
            hh["vs_ladder_level"] = {
                "source": lv.get("source"), "L_dut": lv["L_dut"],
                "rel_L": abs(hh["L_in_core"] - lv["L_dut"]) / abs(lv["L_dut"]),
                "rel_grad_l2": float(np.linalg.norm(gh_ - gl) / np.linalg.norm(gl)),
                "note": "the ladder's level, measured on the RTX 4090 in another job, against "
                        "this lane's in-memory run on the A6000: the same protocol on a "
                        "different card, i.e. a second floor for H1's tolerance"}
        hh["device_peak_note"] = (
            "the sampled device high-water mark does not separate the two modes: "
            f"{hh['device_baseline_used_gb'].get('hybrid')} GB of the "
            f"{hh['device_peak_used_gb'].get('hybrid')} GB peak was already held before the "
            "hybrid run began (cudaMemGetInfo counts what the previous factorisation's "
            "allocator kept, and cupy's pool held only "
            f"{((pr.get('hybrid') or {}).get('device_peak_pool_total_gb'))} GB of it). "
            "The evidence that the factor moved is cuDSS's plan (device memory = the limit) "
            "and the process's resident set.")
    elif h1:
        out["h1"] = {"level": h1.get("level"), "error": h1.get("error")}
    # REPLICATES. H1's number is a DIFFERENCE between two runs of a solver
    # that is not bit-reproducible, so how much that difference itself moves
    # from job to job is part of reading it -- every P3 job that ran the
    # comparison is listed, with the provenance of its plan numbers
    reps = []
    for d2, r2 in all_runs("p3", "p3_ladder.json"):
        h2 = r2.get("h1") or {}
        g2 = h2.get("gate") or {}
        if not g2:
            continue
        reps.append({
            "run": d2.name, "level": h2.get("level"), "limit": h2.get("limit"),
            "rel_L": g2.get("rel_L"), "rel_grad_l2": g2.get("rel_grad_l2"),
            "rel_grad_worst": g2.get("rel_grad_worst"),
            "control_in_core_repeat_rel_grad_l2": (
                g2.get("control_in_core_repeat") or {}).get("rel_grad_l2"),
            "plan_limit_applied": ((h2.get("plans") or {}).get("hybrid")
                                   or {}).get("hybrid_limit_applied"),
            "slowdown_value_and_grad": g2.get("slowdown_value_and_grad")})
    if reps:
        gl2 = [v["rel_grad_l2"] for v in reps if v.get("rel_grad_l2")]
        gw = [v["rel_grad_worst"] for v in reps if v.get("rel_grad_worst")]
        out["h1_replicates"] = {
            "runs": reps,
            "rel_grad_l2_range": [min(gl2), max(gl2)] if gl2 else None,
            "rel_grad_l2_spread": (max(gl2) / min(gl2)) if len(gl2) > 1 else None,
            "rel_grad_worst_range": [min(gw), max(gw)] if gw else None,
            "rel_grad_worst_spread": (max(gw) / min(gw)) if len(gw) > 1 else None,
            "note": "the same protocol, the same card, different jobs: the "
                    "hybrid-vs-in-core gradient difference moves by this factor "
                    "between jobs, so the single number gate H1 reports is not "
                    "reproducible to better than that either. Both ends are "
                    "above the 1e-9 the gate asks of it in the runs that are "
                    "listed with a control, and the in-core-vs-in-core control "
                    "of each run is the floor under both."}
    m5 = r.get("m5_feasibility") or {}
    if m5:
        pl = m5.get("plans") or {}
        host = m5.get("host") or {}
        out["m5"] = {
            "level": m5.get("level"), "n_unknowns": (m5.get("grid") or {}).get("n_unknowns"),
            "device_memory_limit": m5.get("limit"), "device_memory_limit_gb": m5.get("limit_gb"),
            "in_core_factor_gb": m5.get("in_core_factor_gb"),
            "hybrid_plan_host_estimate_gb": m5.get("hybrid_plan_host_estimate_gb"),
            # cuDSS reports a different hybrid MINIMUM depending on whether
            # hybrid mode was on when it planned; both are recorded, and the
            # blocker (host memory) is the same under either
            "hybrid_min_device_memory_gb": m5.get("hybrid_min_device_memory_gb"),
            "hybrid_min_device_memory_gb_in_core_plan": m5.get(
                "hybrid_min_device_memory_gb_in_core_plan"),
            "plan_limit_applied": (pl.get("hybrid") or {}).get("hybrid_limit_applied"),
            "plan_seconds": {k: (pl.get(k) or {}).get("plan_seconds")
                             for k in ("in_core", "hybrid")},
            "host_needed_gb": m5.get("host_needed_gb"),
            "host_in_use_gb": m5.get("host_in_use_gb"),
            "host_allowed_gb": m5.get("host_allowed_gb"),
            "host_margin_gb": m5.get("margin_gb"),
            "cgroup_limit_gb": (host.get("cgroup") or {}).get("limit_gb"),
            "node_mem_total_gb": (host.get("meminfo") or {}).get("MemTotal"),
            "preset": "gpu-a6000-1",
            "fits": m5.get("fits"), "not_run": m5.get("not_run"), "error": m5.get("error")}
        # what hybrid mode can hold AT ALL on this preset, against the device it
        # is supposed to relieve: the factor lives in host memory, so the
        # container's memory limit is the ceiling
        allowed, used = m5.get("host_allowed_gb"), m5.get("host_in_use_gb")
        dev = ((r.get("env") or {}).get("device_memory_gb") or {}).get("total")
        if allowed is not None and used is not None:
            ceiling = allowed - used - (m5.get("margin_gb") or 0.0)
            out["m5"]["hybrid_ceiling_factor_gb"] = ceiling
            out["m5"]["device_total_gb"] = dev
            out["m5"]["hybrid_reaches_past_the_device"] = bool(dev is not None and ceiling > dev)
    return out


def corrections(study: dict[str, Any]) -> dict[str, Any]:
    """The measured fixture / protocol corrections, stated SEPARATELY from
    the primary V1 number (each is a measurement with its own uncertainty,
    not part of the discretisation study): the 10 W walls (the wall ladder),
    the metal resistance's RC contamination (the RC twin), and the short
    standard's post short (the thru probe, extrapolated like the ladder)."""
    ref = study["referee"]["total"]
    out: dict[str, Any] = {}
    w = study.get("walls") or {}
    if w.get("margins_w") and WALL_W in w["margins_w"]:
        i = w["margins_w"].index(WALL_W)
        out["walls"] = {"what": "L(farthest walls) / L(chosen walls) - 1 at level 2",
                        "sign": "raises L", "rel": w["L_dut"][-1] / w["L_dut"][i] - 1.0,
                        "from_margin_w": WALL_W, "to_margin_w": w["margins_w"][-1]}
    tw = study.get("family_rc_twin") or {}
    if tw.get("rel_to_primary"):
        out["rc"] = {"what": "L(10 MHz, 2e7 S/m) / L(100 MHz, 2e6 S/m) - 1 at the finest level "
                             "with a twin: same skin depth, a tenth of the resistance",
                     "sign": "raises L", "rel": tw["rel_to_primary"][-1],
                     "per_level": dict(zip([str(m) for m in tw["m"]], tw["rel_to_primary"]))}
    th = (study.get("thru") or {}).get("levels") or {}
    ks = sorted((int(k), v["L_deembedded"]) for k, v in th.items() if v.get("L_deembedded"))
    if len(ks) >= 2:
        rt = richardson([WIDTH / k for k, _ in ks], [v for _, v in ks])
        rng = rt["range"]
        out["post_short"] = {
            "what": "the thru probe (a DUT identical to the short's bridge) de-embeds to "
                    "L_thru > 0: the grounded post shorts part of the bridge, so the fixture "
                    "subtracts less than a whole bridge; L_dut - L_thru is what the referee "
                    "computes", "sign": "lowers L",
            "per_level": {str(k): v for k, v in ks},
            "extrapolated_range": rng, "observed_order": rt["observed_order"],
            "rel_range": [-rng[1] / ref, -rng[0] / ref]}
    fam = study.get("family") or {}
    rr = (fam.get("richardson") or {}).get("range")
    if rr and out:
        lo, hi = rr
        f_lo = f_hi = 1.0
        for k in ("walls", "rc"):
            if k in out:
                f_lo *= 1.0 + out[k]["rel"]
                f_hi *= 1.0 + out[k]["rel"]
        lo, hi = lo * f_lo, hi * f_hi
        if "post_short" in out:
            lo -= out["post_short"]["extrapolated_range"][1]
            hi -= out["post_short"]["extrapolated_range"][0]
        out["all"] = {"L_range": [lo, hi], "rel_range": [lo / ref - 1.0, hi / ref - 1.0]}
    return out


def resolution_needed(study: dict[str, Any], target: float = V1_TIGHT) -> dict[str, Any]:
    """What resolution a ``target`` (3 %) absolute L needs, from the fitted
    error model ``L(m) = L_inf - C (W/m)^p`` of the finest three levels (the
    observed order; p = 1 and p = 2 bracket it): the smallest m whose
    discretisation error ``C (W/m)^p`` is below ``target`` of the limit, and
    whether the limit itself is within ``target`` of the referee."""
    fam = study.get("family") or {}
    r = fam.get("richardson") or {}
    ms, L = fam.get("m") or [], fam.get("L_dut") or []
    if len(ms) < 3:
        return {}
    ref = study["referee"]["total"]
    out: dict[str, Any] = {"target": target, "per_order": {}}
    orders = {"p1": 1.0, "p2": 2.0}
    if r.get("observed_order"):
        orders["observed"] = float(r["observed_order"])
    ests = r.get("estimates") or {}
    for name, p in orders.items():
        linf = ests.get({"p1": "p1_finest_pair", "p2": "p2_finest_pair"}.get(name, "observed_order"))
        if linf is None:
            continue
        c = (linf - L[-1]) / (1.0 / ms[-1]) ** p          # in units of W^p
        # e(m) = c m^-p / L_inf <= target  ->  m >= (c / (target L_inf))^(1/p)
        m_need = (c / (target * linf)) ** (1.0 / p) if c > 0 else 1.0
        err = c * ms[-1] ** -p / linf
        out["per_order"][name] = {"order": p, "L_inf": linf, "C": c,
                                  "error_at_finest": err,
                                  "finest_within_target": bool(err <= target),
                                  "cells_across_width_needed": m_need,
                                  "first_level_within_target": int(np.ceil(m_need - 1e-9)),
                                  "limit_vs_referee": linf / ref - 1.0}
    vals = [v["cells_across_width_needed"] for v in out["per_order"].values()]
    out["cells_across_width_needed_range"] = [min(vals), max(vals)]
    out["finest_level"] = int(ms[-1])
    # the level that is under ``target`` whichever order in [1, 2] holds
    out["first_level_within_target_every_order"] = max(
        v["first_level_within_target"] for v in out["per_order"].values())
    out["finest_within_target_every_order"] = bool(
        all(v["finest_within_target"] for v in out["per_order"].values()))
    return out


def extrapolation_checks(study: dict[str, Any]) -> dict[str, Any]:
    """Is the P3 range [p = 2, p = 1 of the finest pair] a bracket of the
    limit? Three measured checks, reported (no gate): (i) the observed order
    of every level triple and the p = 1 / p = 2 estimate of every level pair
    (does the bracket close in from both sides?); (ii) a least-squares fit
    ``L = L_inf + a h + b h^2`` over all levels; (iii) the exact cubic
    ``L_inf + a h + b h^2 + c h^3`` through four levels. Each limit against
    the referee and whether it falls inside the P3 range."""
    fam = study.get("family") or {}
    ms, L = fam.get("m") or [], fam.get("L_dut") or []
    rng = (fam.get("richardson") or {}).get("range")
    if len(ms) < 3 or not rng:
        return {}
    ref = study["referee"]["total"]
    h = np.asarray([1.0 / m for m in ms])
    v = np.asarray(L, dtype=np.float64)
    out: dict[str, Any] = {"triples": {}, "pairs": {}, "fits": {}}
    for i in range(len(ms) - 2):
        rr = richardson(list(h[i:i + 3] * WIDTH), list(v[i:i + 3]))
        out["triples"][",".join(str(m) for m in ms[i:i + 3])] = {
            "observed_order": rr["observed_order"],
            "estimate": (rr["estimates"] or {}).get("observed_order")}
    for i in range(len(ms) - 1):
        rr = richardson(list(h[i:i + 2] * WIDTH), list(v[i:i + 2]))
        e = rr["estimates"]
        out["pairs"][",".join(str(m) for m in ms[i:i + 2])] = {
            "p1": e["p1_finest_pair"], "p2": e["p2_finest_pair"]}
    orders = [t["observed_order"] for t in out["triples"].values()]
    out["observed_order_rising"] = bool(all(o is not None for o in orders)
                                        and all(b > a for a, b in zip(orders[:-1], orders[1:])))
    p1s = [x["p1"] for x in out["pairs"].values()]
    p2s = [x["p2"] for x in out["pairs"].values()]
    # p = 1 pair estimates falling and p = 2 ones rising, pair after pair
    out["bracket_closes_from_both_sides"] = bool(
        len(p1s) >= 2 and all(b < a for a, b in zip(p1s[:-1], p1s[1:]))
        and all(b > a for a, b in zip(p2s[:-1], p2s[1:])))
    fits: dict[str, Any] = {}
    A = np.vstack([np.ones_like(h), h, h ** 2]).T
    c_lsq, *_ = np.linalg.lstsq(A, v, rcond=None)
    fits["lsq_h_h2_all_levels"] = float(c_lsq[0])
    if len(ms) >= 4:
        hh, vv = h[-4:], v[-4:]
        A4 = np.vstack([np.ones_like(hh), hh, hh ** 2, hh ** 3]).T
        fits["exact_cubic_finest_four"] = float(np.linalg.solve(A4, vv)[0])
    for k, lim in fits.items():
        out["fits"][k] = {"L_inf": lim, "rel_to_referee": lim / ref - 1.0,
                          "inside_p3_range": bool(rng[0] <= lim <= rng[1])}
    return out


def residual_block(study: dict[str, Any]) -> dict[str, Any]:
    """The gap between the extrapolated limit and the referee, against
    EVERY model difference this study measured between the FDFD fixture and
    the referee: the signed ones of ``corrections`` (walls, RC, post short)
    and the two referee-convention bounds (the omitted via, the M2 -> M1
    transition across the post). (The underpass one layer up / down is not
    one: the referee's underpass sits at the fixture's own M1.) Whether the
    residual is explained, and how its size compares with the extrapolation's
    own p-spread and with every extrapolation estimate recorded."""
    fam = study.get("family") or {}
    r = fam.get("richardson") or {}
    rng = r.get("range")
    cor = study.get("corrections") or {}
    ref_b = study.get("referee") or {}
    if not rng or "all" not in cor:
        return {}
    ref = ref_b["total"]
    signed = {k: cor[k]["rel"] for k in ("walls", "rc") if k in cor}
    ps = (cor.get("post_short") or {}).get("rel_range")
    bounds = {"via": (ref_b.get("via") or {}).get("bound_rel"),
              "post_split_band": ref_b.get("post_split_band_rel")}
    mags = [abs(x) for x in signed.values()] + ([max(abs(x) for x in ps)] if ps else []) \
        + [abs(x) for x in bounds.values() if x is not None]
    total = float(sum(mags))
    after = cor["all"]["rel_range"]
    bsum = float(sum(abs(x) for x in bounds.values() if x is not None))
    # the finest-level estimates: the finest pair at p = 1 / 2, the observed
    # order of the finest three, and the two whole-ladder fits
    ests = list((r.get("estimates") or {}).values())
    ests += [f["L_inf"] for f in ((study.get("extrapolation_checks") or {}).get("fits") or {}).values()]
    closest = min(abs(e / ref - 1.0) for e in ests)
    out = {"limit_rel_range": [v / ref - 1.0 for v in rng],
           "signed_corrections_rel": signed, "post_short_rel_range": ps,
           "referee_convention_bounds_rel": bounds,
           "measured_model_differences_abs_sum": total,
           "after_corrections_rel_range": after,
           "unexplained_at_least": float(min(abs(x) for x in after) - bsum),
           "richardson_p_spread_rel": (rng[1] - rng[0]) / ref,
           "finest_estimates_below_referee_by_at_least": float(closest),
           # explained only if every measured difference, at its worst and all
           # pulling the same way, could close the nearest end of the range
           "explained": bool(total >= min(abs(v / ref - 1.0) for v in rng))}
    return out


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--from-json", action="store_true",
                    help="reuse the referee and the P1 blocks of the JSON on disk; re-read the "
                         "harvested lanes and recompute everything derived")
    ap.add_argument("--no-figure", action="store_true")
    args = ap.parse_args(argv)
    import jax
    jax.config.update("jax_enable_x64", True)
    reuse = json.loads(JSON_PATH.read_text()) if (args.from_json and JSON_PATH.exists()) else {}
    study = assemble(reuse)
    JSON_PATH.write_text(json.dumps(study, indent=1))
    print(f"wrote {JSON_PATH}")
    for k, v in study["gates"].items():
        print(f"  {k}: {'PASS' if v.get('passed') else 'FAIL'}")
    if not args.no_figure:
        figure(study, PNG_PATH)
        print(f"wrote {PNG_PATH}")
    return 0


# ----------------------------------------------------------------------------
# 7. figure

C_BLUE, C_ORANGE, C_AQUA, C_VIOLET, C_RED = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7", "#e34948"
C_INK, C_INK2, C_MUTED, C_SURF = "#0b0b0b", "#52514e", "#a3a29c", "#fcfcfb"


def figure(study: dict[str, Any], path: pathlib.Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker

    plt.rcParams.update({"font.size": 9, "axes.edgecolor": C_MUTED, "axes.labelcolor": C_INK2,
                         "xtick.color": C_INK2, "ytick.color": C_INK2, "axes.titlesize": 10,
                         "axes.titlecolor": C_INK, "axes.grid": True, "grid.color": "#e6e5e0",
                         "grid.linewidth": 0.6, "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, ax = plt.subplots(2, 3, figsize=(15.0, 8.4), facecolor=C_SURF)
    for a in ax.ravel():
        a.set_facecolor(C_SURF)
    ref = study["referee"]["total"] * 1e12

    # (a) the ladder
    a = ax[0, 0]
    a.axhline(ref, color=C_INK, lw=1.2)
    a.axhspan(ref * 0.95, ref * 1.05, color=C_MUTED, alpha=0.12, lw=0)
    a.axhspan(ref * 0.97, ref * 1.03, color=C_MUTED, alpha=0.18, lw=0)
    a.text(0.02, ref * 1.004, f"referee {ref:.1f} pH (bands +-3 %, +-5 %)", color=C_INK,
           fontsize=8, transform=a.get_yaxis_transform())
    fam, alt = study.get("family") or {}, study.get("family_alt") or {}
    try:
        old = json.loads((HERE / "gpu_scaling.json").read_text())["extended"]
        a.plot(np.asarray(old["cells_across_width"]) ** -1, np.asarray(old["L_dut"]) * 1e12,
               "^--", color=C_MUTED, ms=6, lw=1.2, label="old cell-defined fixture (W/1..W/6)")
    except Exception:
        pass
    if alt.get("m"):
        a.plot(1.0 / np.asarray(alt["m"]), np.asarray(alt["L_dut"]) * 1e12, "s-", color=C_ORANGE,
               ms=7, lw=1.6, label="invariant, metal_cells 2 at level 1")
    if fam.get("m"):
        a.plot(1.0 / np.asarray(fam["m"]), np.asarray(fam["L_dut"]) * 1e12, "o-", color=C_BLUE,
               ms=8, lw=2.0, label="invariant, metal_cells 1 at level 1 (primary)")
        rr = (fam.get("richardson") or {}).get("range")
        if rr:
            a.fill_between([0, 0.06], [rr[0] * 1e12] * 2, [rr[1] * 1e12] * 2, color=C_BLUE,
                           alpha=0.25, lw=0, label="Richardson range (finest three)")
    a.set_xlim(0, 1.05)
    a.set_xlabel("h / W = 1 / m")
    a.set_ylabel("de-embedded L_dut (pH)")
    a.set_title("(a) joint nested ladder vs the referee")
    a.legend(loc="lower left", fontsize=7, frameon=False)

    # (b) P6
    a = ax[0, 1]
    p6 = (study.get("p6") or {}).get("levels") or {}
    names = [("a_walls", "(a) walls"), ("b_short", "(b) short std."), ("c_vertical", "(c) z refined"),
             ("abc", "all three"), ("new", "new fixture")]
    xs = np.arange(len(names))
    for off, (lvl, col) in zip((-0.18, 0.18), (("1", C_AQUA), ("2", C_VIOLET))):
        vals = [100 * p6.get(lvl, {}).get(k, {}).get("rel_to_old", np.nan) for k, _ in names]
        a.bar(xs + off, vals, width=0.34, color=col, label=f"level {lvl}", edgecolor=C_SURF, lw=2)
    a.axhline(0, color=C_INK2, lw=0.8)
    a.set_xticks(xs, [n for _, n in names], fontsize=8)
    a.set_ylabel("L change vs the old fixture (%)")
    a.set_title("(b) P6: one old level-dependence switched at a time")
    a.legend(fontsize=8, frameon=False)

    # (c) dL/dwidth
    a = ax[0, 2]
    if fam.get("m"):
        gw = [v for v in fam["dL_dwidth"]]
        a.plot(fam["m"], np.asarray([np.nan if v is None else v for v in gw]) * 1e6, "o-",
               color=C_BLUE, ms=8, lw=2, label="jax.grad (cuDSS)")
    fd = study.get("fd_width") or {}
    if fd.get("fd") is not None:
        a.plot([fd["level"]], [fd["fd"] * 1e6], "x", color=C_RED, ms=12, mew=2,
               label=f"FD4 at m={fd['level']} (rel {fd['rel']:.1e})")
    a.set_xlabel("level m (cells across W)")
    a.set_ylabel("dL/dwidth (pH/um)")
    a.set_title("(c) shape derivative by level (P4, P5)")
    a.legend(fontsize=8, frameon=False)

    # (d) wall ladder
    a = ax[1, 0]
    w = study.get("walls") or {}
    if w.get("margins_w"):
        lw_ = np.asarray(w["L_dut"]) * 1e12
        a.plot(w["margins_w"], 100 * (lw_ / lw_[-1] - 1.0), "o-", color=C_BLUE, ms=8, lw=2)
        a.axhspan(-100 * WALL_TOL, 100 * WALL_TOL, color=C_MUTED, alpha=0.18, lw=0)
        a.set_xscale("log")
        a.set_xticks(w["margins_w"], [f"{v:g} W" for v in w["margins_w"]])
        a.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    a.set_xlabel("wall / lid margin")
    a.set_ylabel("L_dut vs the 80 W walls (%)")
    a.set_title("(d) wall ladder at level 2 (band: +-0.2 %)")

    # (e) sigma plateau
    a = ax[1, 1]
    pl = study.get("sigma_plateau") or {}
    if pl.get("sigma"):
        a.plot(pl["skin_over_width"], np.asarray(pl["L_dut"]) * 1e12, "o-", color=C_BLUE, ms=8, lw=2,
               label=f"100 MHz, level {pl.get('level')}")
        if pl.get("rc_twin"):
            tw = pl["rc_twin"]
            a.plot([tw["skin_depth"] / WIDTH], [tw["L_dut"] * 1e12], "D", color=C_ORANGE, ms=8,
                   label="10 MHz, 2e7 S/m (same delta)")
        a.axvline(3.0, color=C_INK2, lw=0.8, ls=":")
        a.axvline(skin_depth(FREQ, SIGMA) / WIDTH, color=C_BLUE, lw=0.8, ls="--")
        a.set_xscale("log")
    a.set_xlabel("skin depth / W")
    a.set_ylabel("L_dut (pH)")
    a.set_title("(e) uniform-current plateau (dotted: delta = 3 W)")
    a.legend(fontsize=8, frameon=False)

    # (f) cost
    a = ax[1, 2]
    cost = study.get("cost") or {}
    if cost:
        ks = sorted(cost, key=int)
        n = [cost[k]["n_unknowns"] for k in ks]
        gb = [cost[k].get("device_gb_after_factor") or np.nan for k in ks]
        a.plot(n, gb, "o-", color=C_BLUE, ms=8, lw=2,
               label="solved levels: device GB in use after one factorisation")
        for k, x_, y_ in zip(ks, n, gb):
            a.annotate(f"m={k}", (x_, y_), textcoords="offset points", xytext=(4, -10), fontsize=8,
                       color=C_INK2)
    plans = study.get("plans") or {}
    pn = [(v["n"], v["permanent_device_memory_gb"]) for v in plans.values()
          if v.get("permanent_device_memory_gb")]
    if pn:
        a.plot(*zip(*sorted(pn)), "s", color=C_MUTED, ms=6,
               label="cuDSS plans: permanent factor GB (x-axis: plan N)")
    a.axhline(24, color=C_INK2, lw=0.8, ls=":")
    a.axhline(48, color=C_INK2, lw=0.8, ls="--")
    hb = study.get("hybrid") or {}
    h1 = hb.get("h1") or {}
    if h1.get("device_memory_limit_gb") and (h1.get("plan_permanent_device_gb") or {}):
        nh = h1.get("n_unknowns")
        dev = (h1["plan_permanent_device_gb"] or {}).get("hybrid")
        host = (h1["plan_permanent_host_gb"] or {}).get("hybrid")
        if nh and dev:
            a.plot([nh], [dev], "*", color=C_ORANGE, ms=15,
                   label=f"hybrid at m={h1['level']} (limit "
                         f"{h1['device_memory_limit_gb']:.0f} GB): plan {dev:.1f} GB on the "
                         f"card, {host:.1f} GB on the host")
    m5 = hb.get("m5") or {}
    if m5.get("host_allowed_gb"):
        a.axhline(m5["host_allowed_gb"], color=C_RED, lw=0.9, ls="-.")
        a.text(0.02, m5["host_allowed_gb"] * 1.06,
               f"host memory this preset may hold ({m5['host_allowed_gb']:.0f} GB): "
               "hybrid mode's ceiling", color=C_RED, fontsize=7,
               transform=a.get_yaxis_transform())
    a.set_xscale("log")
    a.set_yscale("log")
    a.set_xlabel("unknowns N")
    a.set_ylabel("memory (GB)")
    a.set_title("(f) cost (24 / 48 GB cards; hybrid mode and its host ceiling)")
    a.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=C_SURF)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
