"""Study H: an RFIC-scale spiral inductor through the whole differentiable
FDFD pipeline -- absolute L against an independent referee, lossy metrics at
2.45 GHz, validated shape derivatives, and a gradient design loop -- on a
geometry and a stack taken from a published 2.4 GHz LC-VCO.

WHAT IS AND IS NOT THE PAPER
----------------------------
The geometry numbers come from the 2.4 GHz LC-VCO paper (arXiv 2607.08852,
IHP SG13G2), which optimised a 3-turn SYMMETRIC OCTAGONAL spiral with
``r_out = 218 um``, ``W = 30 um``, ``S = 14 um`` on TopMetal2 with a
TopMetal1 underpass and reported ``L_diff = 4.000 nH``, ``Q_diff = 16.80``
at 2.45 GHz.

THIS STUDY SOLVES A SQUARE SPIRAL, NOT THE PAPER'S OCTAGON. The pipeline's
generator is :func:`rfx.fdfd.gds.rect_spiral` (an octagonal generator exists
but the differentiable body-fitted metric of :mod:`rfx.fdfd.spiral` is built
on axis-aligned breakpoints and takes the rectangular one only). A square
spiral of the same ``r_out`` has MORE metal and a longer conductor than the
octagon that fits inside it -- an octagon cuts the four corners -- so its
inductance is HIGHER at equal ``r_out``, and it is a different device, not a
discretisation of the same one. Moreover the paper's inductor is SYMMETRIC
(two interleaved windings, differentially driven), while this one is
single-ended with an underpass return. The paper's 4.000 nH / 16.80 are
therefore CONTEXT ONLY: they are quoted to say where the geometry came from
and to give the design loop of part C a target of the right order. NO GATE
IN THIS STUDY COMPARES ANYTHING TO THEM. Every accuracy gate is against the
Greenhouse partial-inductance referee
(``validation/crossval/comparators/spiral_greenhouse.py``) evaluated on the
SAME square geometry, under the protocol validated in
``validation/fdfd/spiral_convergence.py`` (study D2).

THE STACK (SG13G2-like, through ``spiral.SmallStack``)
------------------------------------------------------
Bottom up: PEC ground at z = 0, silicon (eps 11.9, ``sigma_si = 2`` S/m in
part B), oxide (eps 4.1), TopMetal1 2 um, TopVia2, TopMetal2 3 um
(sigma 3.05e7 S/m), passivation, air, PEC lid. The numbers follow
:func:`rfx.fdfd.gds.sg13g2_stack` (itself approximate public numbers, not
PDK data); ``SmallStack`` has one oxide permittivity for the whole back end,
so the passivation's eps 6.6 is lumped into the oxide's 4.1, and its via
slab is widened from the public ~0.85 um TopVia2 to 2 um so that the
thinnest metal in the stack -- the quantity
:func:`rfx.fdfd.spiral.leontovich_validity` divides the skin depth by -- is
the 2 um underpass and not the via. Both are documented deviations, listed
with their consequences in ``meta.deviations`` of the JSON.

SUBSTRATE THICKNESS: THE BIGGEST DELIBERATE DEVIATION
-----------------------------------------------------
The real die is ~190 um of silicon over a backside ground. The fixture puts
a PEC ground plane under the silicon and resolves the silicon with the
LAYER-uniform vertical grid of :func:`rfx.fdfd.gds.z_lines` (``base_dz``
caps the cell size inside each layer; there is no geometric grading inside a
layer -- the grading available here is BETWEEN layers, 60 um cells in the
silicon and the air against 1.0-1.5 um cells in the metals, plus the
geometrically graded ``pad_cells_z`` cells that carry the PEC lid away).
Every silicon cell is 3 x 43 x 44 unknowns, so the thickness is bought in
whole cells. This study runs ``t_si = 120 um`` (two 60 um cells) and
measures the sensitivity against ``t_si = 60 um`` (one cell) at the coarse
level, at 2.45 GHz, on L AND Q. The referee gives the same sensitivity
analytically and for free (the ground image moves), at 60 / 120 / 190 /
300 um, so the FDFD's two points can be read against a curve rather than
extrapolated blindly.

THE RESOLUTION FLOOR: ``base_dx`` IS NOT A FREE KNOB AT THIS PITCH
------------------------------------------------------------------
``SpiralSpec.base_dx`` caps the cell size only where the geometry leaves it
free. :func:`rfx.fdfd.gds.mesh_lines` makes every polygon edge a grid line
and then grades between them with neighbour ratio <= 1.5, and this geometry
has a 14 um turn-to-turn gap next to a 30 um strip: the mandatory lines
14 um apart force ~14 um cells, and the ratio-1.5 rule then forbids the
neighbouring 30 um strip from being one cell. The measured consequence is
that ``base_dx = W`` (30 um) and ``base_dx = W/2`` (15 um) are the SAME
GRID to within one line (nx 43 vs 42, N 85785 vs 83820) and both put TWO
cells across the strip width. The next level the mesher will actually give
is ``base_dx = W/3`` (10 um), which lands on 5 um cells -- six across the
strip -- and N = 509580, far outside this machine's budget. So the two
levels of part A are the same effective resolution, which is exactly the
small spiral's W/2 level in D2, and that is the level whose bias D2
measured as -12.0 %. The comparison in gate R1 is apples to apples for that
reason and for no other.

PARTS
-----
A. Absolute L at 100 MHz under D2's validated protocol (uniform-current
   volumetric metal ``sigma = 3e6``, vacuum dielectrics, ``pad_cells = 4``,
   area-exact corners) at ``base_dx`` = W and W/2, against the PRIMARY
   referee convention: what the open/short fixture measures is
   ``L(strip) - L(short's bridge)``, with the bridge built by
   ``spiral_convergence.bridge_segments`` (imported and reused, its
   module-level fixture constants rebound to this geometry -- see
   :func:`bind_referee`). ``spiral_greenhouse.mohan_wheeler`` is reported as
   a secondary closed-form sanity number for the square spiral.
B. 2.45 GHz with a Leontovich metal and lossy silicon and the real
   permittivities: L_diff, Q_diff, L_se, Q_se at both levels; a frequency
   sweep; reciprocity and passivity of every solved S-matrix; the side-wall
   independence gate; ``jax.grad`` of L_diff and of Q_diff against a 4th
   order central difference.
C. A resumable L-BFGS-B design loop at the coarse level: hit
   ``L_diff = 4.000 nH`` within 2 % while maximising Q_diff over
   (r_out, spacing, width) in a feasible box around the paper's values,
   with the gradient at the optimum checked against FD4.

WHAT CAME OUT (every number here is in the JSON on disk)
--------------------------------------------------------
A, absolute L at 100 MHz. ``base_dx = W`` and ``W/2`` are the same grid, as
promised above, and they agree: N = 85785 / 83820, in-plane cells 13-15 um,
TWO across the strip both times, ``L_diff`` = 3089.74 / 3088.30 pH, 0.047 %
apart. The referee's primary quantity is 3560.37 pH (strip 3623.94 pH minus
the short standard's bridge 63.57 pH, 1.75 % of the strip), so the FDFD is
-13.218 % / -13.259 % LOW. D2's small spiral at the SAME two cells across
the strip was -12.025 %: the RFIC-scale deficit is the same strip
cross-section discretisation bias, 1.234 points deeper on a device four
times the size with three turns instead of two (gate R1, tolerance 3
points). Pushing the four side walls out (``pad_cells`` 4 -> 8, walls 431 ->
1356 um, N = 119689) under the same protocol raises L to 3121.04 pH, gap
-12.339 %, which is 0.314 points from D2's number -- most of the extra
deficit was the box, not the strip. The referee itself is corroborated
independently: Mohan's modified-Wheeler closed form gives 4.1653 nH against
the referee's free-space strip 4.0243 nH, +3.50 %, inside Mohan's own
2-8 % band. ``base_dx = W/3`` (5 um cells, six across the strip) would be
N = 509580 and was built but not solved.

B, 2.45 GHz. ``skin_depth`` = 1.8411 um for the stack's 3.05e7 S/m, so
``leontovich_validity`` (skin depth over the THINNEST metal slab, the 2 um
TopMetal1 underpass) = 0.9206 and delta / t(TopMetal2, 3 um) = 0.6137; PURE
copper at 5.8e7 S/m would give 0.4450 on the 3 um metal, which is the ~0.45
the sheet model is usually quoted at. At 0.61-0.92 the sheet's ``delta << t``
premise is violated: the absolute Q below is NOT a validated number.
``L_diff`` / ``Q_diff`` / ``L_se`` / ``Q_se`` = 3043.33 pH / 20.6454 /
3196.97 pH / 13.2030 at W/1 and 3042.76 pH / 20.6151 / 3196.24 pH / 13.1954
at W/2 (0.019 % and 0.147 % apart -- the same grid again). The sweep at W/1,
(f in GHz: L_diff in pH, Q_diff): (0.5: 3041.3, 14.692), (1: 2991.6,
18.106), (2.45: 3043.3, 20.645), (4: 3270.1, 18.816), (6: 3915.4, 14.123).
L is NOT monotone -- it dips 1.6 % at 1 GHz -- because two things move at
once: the Leontovich sheet's internal inductance falls like
1/(sigma delta omega) while the self-resonant uplift rises. Fitting
L = L0 / (1 - (f/f0)^2) to the 2.45 and 6 GHz points puts the first
self-resonance at f0 = 11.9 GHz, and Q peaks between 2.45 and 4 GHz. Every
solved S-matrix is reciprocal to 1.97e-10 and passive to 1.0000000001
(12 solves; the de-embedded S, a post-processing construct, is itself
passive to 0.99999998). The side-wall gate FAILS, marginally and
informatively: pad 4 -> 8 moves ``L_diff`` by +1.0245 % against a < 1 %
criterion, but the steps are +0.983 % (4 -> 6) then +0.041 % (6 -> 8), so
the box is converged from pad 6 on and the pad-4 grid that parts A, B and C
all use is 1.02 % low from the side walls alone. Substrate thickness
120 -> 60 um moves L by -16.096 % and Q by -6.258 %, against the referee's
ground image -15.290 % for the same pair; the referee puts the chosen 120 um
6.227 % below the real 190 um die. Gradients at the nominal:
``dL/d(r_out, spacing, width)`` = (3.0691e-5, -6.0777e-5, -1.0061e-4) H/m
and ``dQ/d(...)`` = (-368.24, -15755.7, -27415.2) 1/m. Against FD4 of the
same discrete model with 1 % steps: ``dL/dr_out`` 1.40e-8, ``dQ/dwidth``
5.75e-8 (gate R3, tolerance 1e-4), and for free from the same stencils
``dL/dwidth`` 5.67e-10 and ``dQ/dr_out`` 4.52e-5. Against the INDEPENDENT
referee's own FD4 gradient (3.0619e-5, -6.2437e-5, -9.6886e-5) the FDFD's
AD gradient agrees to +0.24 %, -2.66 %, +3.85 % per component -- an order
of magnitude better than the 13 % it is wrong by in absolute value, and
better than the 5-15 % D2 measured on the small spiral.

C, the design loop. The penalty weight had to be MEASURED, and the first
attempt is kept in the JSON as ``C_attempt_1`` because it is that
measurement. Sizing lam by the rule of ``spiral_design.py`` -- from the
gradient ratio at the NOMINAL theta, for a 1 % allowed L error -- gives
lam = 1.1213, and at that weight the loop reduced the loss monotonically
while moving AWAY from the target: 1.72 % L error at its fifth evaluation,
6.35 % at its seventh, buying 1.1 % of Q for it. That is what a soft
constraint does when its weight is too small, and it also MEASURES the
trade: d(Q/Q_ref)/d|e| = 0.0112/0.0463 = 0.242 in loss units, so the
stationary error under a weight lam is 0.242/(2 lam) and lam = 30 pins it at
0.40 %. Re-run at lam = 30 (``--lam 30``), L-BFGS-B converged on its
projected gradient in 9 iterations and 14 objective evaluations -- 14
forward three-fixture solves and 14 adjoint passes, 10 of the evaluations
accepted as iterates, 0 replay mismatches across 14 chunked processes --
to theta* = (r_out 211.031, spacing 10.320, width 22.000) um with
``L_diff`` = 3989.62 pH, 0.2596 % from the 4.000 nH target (gate R4, band
2 %), and ``Q_diff`` = 20.5771 against 20.6454 at the nominal: the target
was met for a 0.33 % Q cost. ``width`` ends ON its lower bound and
``spacing`` 0.32 um above its own, which is why the FD4 check at the
optimum uses the one-sided 4th-order stencil for ``width`` -- a
bound-constrained optimum cannot be probed symmetrically.

And the honest reading of that design: it is the optimum OF A MODEL THAT IS
12.3 % LOW. Divide the delivered ``L_diff`` by the wall-corrected bias of
part A and the same geometry would really be about 4.55 nH, 13.8 % above the
target. The loop found the right SHAPE change -- its derivatives are right
to 1e-8 against its own model and to a few per cent against the independent
referee -- on a model whose absolute value is not.

Cost (gate R5). Largest N solved 119689; per-fixture factorisation + solve
44.6-134.0 s (median 76.7) on the study's levels; peak physical footprint
8.92 GB through ``solve_split`` and 12.44 GB through the single
``solve_spiral`` call the FD stencils use, peak psutil RSS 7.85 GB. The cost
LAW of this fixture family, measured on five single-fixture solves at
N = 45744..85785 (21.9 -> 79.1 s, 2.15 -> 7.41 GB peak footprint, N moved by
the wall padding alone): seconds ~ N^2.04 and peak footprint ~ N^2.12.

RUNNING IT (it does not fit in one process)
-------------------------------------------
One three-fixture solve at this scale measured 218-249 s at N ~ 85000 and
384-386 s at N = 119689, and one ``value_and_grad`` 466-476 s with one LU
factor kept, and the study needs tens of them, so every block is resumable:
the JSON is rewritten after every solve, ``--from-json`` reassembles,
``--max-solves K`` stops cleanly after K solve units and exits with code 3
("more to do"), and the design loop resumes by REPLAYING its logged
evaluations into a fresh ``minimize`` (the pattern of
``validation/fdfd/spiral_design.py``, with the same bitwise replay check).
Items that do not depend on each other can also be split ACROSS processes
(``--only`` for part B, ``--a-items``, ``--c-items``, ``--fd-params``) into
separate JSON files and folded back with ``--merge``; this run used that for
the FD stencils, for part A's wall solve and for the design loop, and the
sibling runs are listed in ``merged_runs``. Drive it in chunks:

    while true; do
      .venv/bin/python validation/fdfd/rfic_spiral.py --from-json --max-solves 1
      [ $? -eq 3 ] || break
    done

Memory: ``--factor-cache`` (``linear_solve.factor_cache_size``, 1 here by
default) is what made these sizes fit, not a tuning choice. With the stock
cache of 4 the three fixtures' LU factors are all live at once and a single
N = 85785 three-fixture solve measured a 15.29 GB physical footprint on this
machine (216.7 s, recorded in ``meta.calibration``); with ONE factor kept
the same solve peaks at 7.41-8.92 GB. The price is that a reverse pass
re-factorises the fixtures whose factor has been evicted -- five
factorisations per gradient instead of three, measured 466-476 s against a
218-249 s forward solve -- so the design loop of part C was run with
``--factor-cache 3`` once the rest of the study had finished and the machine
was free; that changes cost only, since the solves are deterministic.

SCOPE FENCE
-----------
Closed padded PEC box, no PML (D2 measured the PML useless at 100 MHz).
Staircase metal with either a uniform-current volumetric conductivity (part
A, the referee's own model) or a Leontovich surface sheet (part B); no
finite-thickness skin effect either way, so part B's ABSOLUTE Q is not a
validated number -- ``leontovich_validity`` is 0.61-0.92 here, i.e. the
sheet model's ``delta << t`` premise is violated and the sheet
over-estimates internal inductance and loss. Part B's gates are
self-consistency (reciprocity, passivity, wall independence) and
derivatives; part A's gate is the absolute L against the referee.
"""
from __future__ import annotations

import argparse
import ctypes
import gc
import importlib.util
import json
import math
import os
import pathlib
import sys
import time
from typing import Any, Callable, Sequence

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
JSON_PATH = HERE / "rfic_spiral.json"
PNG_PATH = HERE / "rfic_spiral.png"
REFEREE_PATH = REPO / "validation" / "crossval" / "comparators" / "spiral_greenhouse.py"
CONV_PATH = HERE / "spiral_convergence.py"

# ---------------------------------------------------------------------------
# the fixture: the paper's geometry as a SQUARE spiral on an SG13G2-like stack

N_TURNS = 3
R_OUT, SPACING, WIDTH = 218e-6, 14e-6, 30e-6      # paper values (octagon -> square)
THETA0 = (R_OUT, SPACING, WIDTH)
LEAD = 1.5 * WIDTH                                 # 45 um, D2's convention
MARGIN = WIDTH                                     # 30 um, D2's convention (margin = W)
PARAMS = ("r_out", "spacing", "width")

T_SI = 120e-6                # chosen substrate thickness (2 x 60 um cells)
T_SI_ALT = 60e-6             # the sensitivity point (1 cell)
T_SI_REAL = 190e-6           # the real die the paper's inductor sits on
T_OX_LOW = 6.04e-6           # silicon surface -> TopMetal1 bottom (sg13g2_stack)
T_M1 = 2.0e-6                # TopMetal1 (the underpass)
T_VIA = 2.0e-6               # TopVia2 widened from ~0.85 um (see the module doc)
T_M2 = 3.0e-6                # TopMetal2 (the spiral strip)
T_OX_HIGH = 1.5e-6           # passivation, eps lumped into the oxide's 4.1
T_AIR = 60e-6
BASE_DZ = 60e-6
METAL_CELLS = 2
PAD_CELLS, PAD_CELLS_Z, PAD_RATIO = 4, 3, 1.5
EPS_SI, EPS_OX = 11.9, 4.1

FREQ_A = 1.0e8               # part A: D2's magnetoquasistatic protocol
SIGMA_VOL = 3.0e6            # part A: D2's uniform-current plateau
FREQ_B = 2.45e9              # part B/C: the paper's measurement frequency
SIGMA_CU = 3.05e7            # SG13G2 TopMetal2
SIGMA_CU_PURE = 5.8e7        # pure copper, for the delta/t cross-check only
SIGMA_SI = 2.0               # p-substrate
SWEEP_FREQS = (0.5e9, 1.0e9, 2.45e9, 4.0e9, 6.0e9)
WALL_PADS = (4, 6, 8)      # the gate is 4 -> 8; 6 makes it a three-point ladder
SCALING_PADS = (0, 1, 2, 3, 4)     # the cost-law family: N moved by the wall padding alone

DIVS = (1.0, 2.0)            # base_dx = WIDTH / div
DIV_W3 = 3.0                 # reported, not run (N = 509580)

# referee geometry derived from the stack
DZ_UNDER = 0.5 * T_M1 + T_VIA + 0.5 * T_M2                   # M2 centre -> M1 centre
def ground_height(t_si: float = T_SI) -> float:
    """PEC ground -> M2 centre line, the referee's image distance."""
    return t_si + T_OX_LOW + T_M1 + T_VIA + 0.5 * T_M2
GROUND_H = ground_height()
REF_SHIFT = 0.5 * WIDTH      # reference plane at the lead-column footprint centre
BRIDGE_SPLIT = 0.5

# part C: the design problem
L_TARGET = 4.000e-9         # the paper's number, as a TARGET (not a gate on physics)
Q_PAPER = 16.80             # the paper's number, context only
L_BAND = 0.02               # gate R4
E_TARGET = 0.01             # the relative L error lam is sized to allow
BOUNDS = ((200e-6, 260e-6), (10e-6, 18e-6), (22e-6, 32e-6))
# the box is the paper values +- roughly 20 %, chosen so that (a) every corner is
# topologically feasible -- the binding margin a_in - width/2 = r_out - width -
# n (width + spacing) is 18 um at the worst corner (200, 18, 32) um against 56 um at
# the nominal -- (b) every breakpoint stays well inside the padded walls at +-431 um,
# and (c) the 4.000 nH target is INTERIOR, so the FD4 check at the optimum has room
# for its +-2 % stencil on every parameter.
MAXITER = 12
GTOL_FACTOR = 1e-3
FD_STEP_REL = 0.01          # 1 % FD steps (D2's, and far above the LU noise floor)

RECIP_TOL = 1e-8
PASSIVITY_TOL = 1.0 + 1e-8
WALL_TOL = 0.01
AD_FD_TOL = 1e-4
R1_POINTS = 0.03            # gate R1: within 3 points of D2's W/2 gap
D2_W2_GAP = -0.12025158857388896      # spiral_convergence.json gates.V1.per_level_gap["W/2"]
D2_GAPS = {"W/1": -0.21202451007372947, "W/2": -0.12025158857388896,
           "W/3": -0.09757369803121485}

EXIT_INCOMPLETE = 3
A_ITEMS = ("levels", "wall")
B_ITEMS = ("leontovich", "levels", "t_si", "sweep", "wall", "grad", "scaling", "fd")
C_ITEMS = ("loop", "fd_opt", "w2")
_ONLY: set[str] = set()          # empty = every item (see ``--only``)
_ONLY_A: set[str] = set()        # ditto for part A (``--a-items``)
_ONLY_C: set[str] = set()        # ditto for part C (``--c-items``)
_FD_PARAMS: set[str] = set()     # which optimum-FD stencils this process does
_BUDGET: dict[str, Any] = {"used": 0, "max": None}
_MEM: dict[str, Any] = {"factor_cache_size": 1}
_C_KEY = "C"                     # where part C's record lives (``--c-key``)
_LAM: float | None = None        # explicit penalty weight (``--lam``)
_T0 = time.time()


def want(item: str) -> bool:
    """Whether part B should do ``item`` in this process. ``--only`` narrows
    it so that two processes can work on DISJOINT items of the same study
    into two JSON files, which are then merged (``--merge``). The items are
    independent by construction -- each is keyed separately under ``B`` and
    none reads another's record except ``t_si`` and ``wall``, which reuse the
    ``levels.1`` solve and are skipped while it is missing."""
    return not _ONLY or item in _ONLY


def want_a(item: str) -> bool:
    """Whether part A should do ``item`` here (``--a-items``)."""
    return not _ONLY_A or item in _ONLY_A


def want_c(item: str) -> bool:
    """Whether part C should do ``item`` here (``--c-items``); the design loop
    itself is strictly sequential, but the FD4 stencil at the optimum and the
    W/2 re-solve are independent of it and of each other once the loop has
    converged, so they can be split across processes and merged."""
    return not _ONLY_C or item in _ONLY_C


def want_fd_param(name: str) -> bool:
    """Whether this process does the optimum-FD stencil of parameter
    ``name`` (``--fd-params``): 4 forward solves each, three parameters,
    trivially splittable."""
    return not _FD_PARAMS or name in _FD_PARAMS


def merge_missing(dst: dict[str, Any], src: dict[str, Any]) -> int:
    """Recursively copy keys of ``src`` that ``dst`` does not have (``dst``
    always wins), returning how many leaves were added. Used by ``--merge``
    to fold a sibling run's disjoint items into this study."""
    added = 0
    for k, v in src.items():
        if k not in dst:
            dst[k] = v
            added += 1
        elif isinstance(dst[k], dict) and isinstance(v, dict):
            added += merge_missing(dst[k], v)
    return added


class BudgetStop(Exception):
    """The solve budget ran out; what is measured is already on disk."""


def spend(n: int = 1) -> None:
    _BUDGET["used"] += int(n)


def check_budget() -> None:
    if _BUDGET["max"] is not None and _BUDGET["used"] >= int(_BUDGET["max"]):
        raise BudgetStop()


# ---------------------------------------------------------------------------
# memory: psutil RSS plus the darwin physical footprint

def proc_memory() -> dict[str, float]:
    """``{rss_gb, footprint_gb, vms_gb}``. ``rss_gb`` is psutil's RSS (the
    number the task asks for); ``footprint_gb`` is
    ``proc_pid_rusage(RUSAGE_INFO_V0).ri_phys_footprint`` on darwin -- what
    Activity Monitor calls Memory, and the number that decides whether this
    machine swaps (RSS misses compressed and swapped anonymous pages). Off
    darwin the footprint is reported as the RSS."""
    out = {"rss_gb": float("nan"), "vms_gb": float("nan"), "footprint_gb": float("nan")}
    try:
        import psutil
        mi = psutil.Process().memory_info()
        out["rss_gb"] = mi.rss / 2.0 ** 30
        out["vms_gb"] = mi.vms / 2.0 ** 30
    except Exception:                            # pragma: no cover - diagnostics only
        pass
    if sys.platform == "darwin":
        try:
            libc = ctypes.CDLL("/usr/lib/libSystem.B.dylib", use_errno=True)
            buf = (ctypes.c_uint8 * 512)()
            if libc.proc_pid_rusage(ctypes.c_int(os.getpid()), ctypes.c_int(0),
                                    ctypes.byref(buf)) == 0:
                v = (ctypes.c_uint64 * 8).from_buffer(buf, 16)
                out["footprint_gb"] = v[7] / 2.0 ** 30
        except Exception:                        # pragma: no cover - diagnostics only
            pass
    else:                                        # pragma: no cover - this machine is darwin
        out["footprint_gb"] = out["rss_gb"]
    return out


def mem_note(tag: str) -> dict[str, Any]:
    m = proc_memory()
    m.update(tag=tag, t=time.time() - _T0, solves=int(_BUDGET["used"]),
             factor_cache_size=int(_MEM["factor_cache_size"]))
    return m


# ---------------------------------------------------------------------------
# 1. the referee (host numpy): D2's construction, rebound to this geometry

def _load(path: pathlib.Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_referee():
    """The Greenhouse comparator (the comparators directory is not a package
    and nothing in ``rfx`` imports it)."""
    return _load(REFEREE_PATH, "spiral_greenhouse")


def bind_referee():
    """``spiral_convergence`` with its module-level FIXTURE CONSTANTS rebound
    to this study's geometry, so that its referee and -- the part that must
    not be reinvented -- its short-standard BRIDGE construction
    (``bridge_segments`` / ``terminal_points``) describe THIS spiral.

    Those functions read ``N_TURNS``, ``LEAD``, ``T_M1``, ``T_M2``,
    ``DZ_UNDER`` as module globals from inside their bodies, so rebinding
    works; ``lead`` and ``ground_height`` are DEFAULT ARGUMENTS, bound at
    def time, so every call here passes them explicitly and this module
    never calls ``cv.referee_value`` / ``cv.bridge_value`` /
    ``cv.referee_deembedded`` (whose signatures cannot carry a ground
    height). What is reused is the segment construction; what is composed
    here is only ``L(strip) - L(bridge)`` at an explicit ground height."""
    cv = _load(CONV_PATH, "spiral_convergence")
    cv.N_TURNS = N_TURNS
    cv.R_OUT, cv.SPACING, cv.WIDTH = R_OUT, SPACING, WIDTH
    cv.LEAD = LEAD
    cv.T_SI, cv.T_OX_LOW = T_SI, T_OX_LOW
    cv.T_M1, cv.T_VIA, cv.T_M2 = T_M1, T_VIA, T_M2
    cv.T_OX_HIGH, cv.T_AIR = T_OX_HIGH, T_AIR
    cv.DZ_UNDER = DZ_UNDER
    cv.GROUND_H = GROUND_H
    cv.THETA0 = THETA0
    cv.REF_SHIFT = REF_SHIFT
    cv.BRIDGE_SPLIT = BRIDGE_SPLIT
    cv.MARGIN = MARGIN
    return cv


def strip_terms(sg, cv, theta: Sequence[float], convention: str = "area_exact",
                shift: float | None = None, ground: float = GROUND_H):
    """Greenhouse terms of the STRIP + underpass, lead shortened to the
    de-embedding reference plane (``shift = None`` -> the lead-column
    footprint centre, which tracks the width)."""
    segs = cv.referee_segments(sg, theta, convention, lead=LEAD, shift=shift)
    return sg.greenhouse_terms(segs, ground_height=ground)


def bridge_terms(sg, cv, theta: Sequence[float], shift: float | None = None,
                 ground: float = GROUND_H, split: float = BRIDGE_SPLIT):
    """Greenhouse terms of the SHORT STANDARD's bridge -- the conductor the
    open/short de-embedding subtracts (D2's construction, reused)."""
    return sg.greenhouse_terms(cv.bridge_segments(sg, theta, shift=shift, split=split),
                               ground_height=ground)


def referee_deembedded(sg, cv, theta: Sequence[float], shift: float | None = None,
                       ground: float = GROUND_H, split: float = BRIDGE_SPLIT,
                       convention: str = "area_exact") -> float:
    """PRIMARY referee quantity: ``L(strip) - L(bridge)``, the inductance the
    FDFD's de-embedded ``L_diff`` actually measures (D2 gate V6 refuted the
    "strip alone" reading directly with a thru probe)."""
    return float(strip_terms(sg, cv, theta, convention, shift, ground).total
                 - bridge_terms(sg, cv, theta, shift, ground, split).total)


def referee_block() -> dict[str, Any]:
    """Everything the referee says about this geometry: the primary value and
    its parts, the reference-plane and bridge-split sensitivities that show
    the DIFFERENCE is the invariant one, the ground-image (substrate
    thickness) curve, the secondary closed form, the corner-convention area
    gate and the FD4 shape gradient."""
    sg, cv = load_referee(), bind_referee()
    st, br = strip_terms(sg, cv, THETA0), bridge_terms(sg, cv, THETA0)
    rec: dict[str, Any] = {
        "what": "Greenhouse partial inductance with a PEC ground image; PRIMARY convention "
                "L(strip) - L(short's bridge), area-exact corners, reference plane at the "
                "lead-column footprint centre (D2's protocol, bridge construction reused)",
        "ground_height": GROUND_H, "dz_underpass": DZ_UNDER, "reference_plane_shift": REF_SHIFT,
        "bridge_split_on_m2": BRIDGE_SPLIT,
        "strip": float(st.total), "bridge": float(br.total),
        "deembedded": referee_deembedded(sg, cv, THETA0),
        "bridge_over_strip": float(br.total / st.total),
        "strip_worst_quadrature_error": float(st.worst_error),
        "bridge_worst_quadrature_error": float(br.worst_error),
        "greenhouse_corner_convention_strip": float(
            strip_terms(sg, cv, THETA0, "greenhouse").total),
        "strip_no_ground": float(strip_terms(sg, cv, THETA0, ground=None).total),
    }
    planes = {"port_line": 0.0, "column_centre": 0.5 * WIDTH, "far_edge": WIDTH}
    rec["per_plane"] = {k: {"strip": float(strip_terms(sg, cv, THETA0, shift=v).total),
                            "bridge": float(bridge_terms(sg, cv, THETA0, shift=v).total),
                            "deembedded": referee_deembedded(sg, cv, THETA0, shift=v)}
                        for k, v in planes.items()}
    vals = [v["deembedded"] for v in rec["per_plane"].values()]
    strips = [v["strip"] for v in rec["per_plane"].values()]
    ref = rec["per_plane"]["column_centre"]["deembedded"]
    rec["reference_plane_spread"] = (max(vals) - min(vals)) / ref
    rec["reference_plane_spread_strip_only"] = (max(strips) - min(strips)) / \
        rec["per_plane"]["column_centre"]["strip"]
    rec["per_split"] = {f"{s:g}": referee_deembedded(sg, cv, THETA0, split=s)
                        for s in (0.0, 0.25, 0.5, 0.75, 1.0)}
    sv = list(rec["per_split"].values())
    rec["split_spread"] = (max(sv) - min(sv)) / ref
    rec["ground_image_curve"] = {
        f"{t * 1e6:g}": {"t_si": t, "ground_height": ground_height(t),
                         "deembedded": referee_deembedded(sg, cv, THETA0, ground=ground_height(t)),
                         "rel_to_chosen": referee_deembedded(
                             sg, cv, THETA0, ground=ground_height(t)) / rec["deembedded"] - 1.0}
        for t in (T_SI_ALT, T_SI, T_SI_REAL, 300e-6)}
    # Mohan's d_in is the inner diameter of the HOLE, i.e. the inner edge of
    # the innermost COMPLETE turn: a0 - (n-1) pitch - W/2 = 100 um here, so
    # d_in = d_out - 2 (n W + (n-1) S) = 200 um. rect_spiral's inner
    # extension (the stub that carries the current to the via) sits inside
    # that hole and is not a turn.
    d_out = 2.0 * R_OUT
    d_in = d_out - 2.0 * (N_TURNS * WIDTH + (N_TURNS - 1) * SPACING)
    d_in_check = 2.0 * (R_OUT - 0.5 * WIDTH - (N_TURNS - 1) * (WIDTH + SPACING) - 0.5 * WIDTH)
    assert abs(d_in - d_in_check) < 1e-15
    rec["mohan_wheeler"] = {
        "what": "SECONDARY sanity number: Mohan's modified-Wheeler fit for a SQUARE spiral "
                "(Mohan et al., JSSC 34(10) 1999, eq. 2; 2-3 % typical and ~8 % worst-case "
                "error against field solvers). The fit has NO ground plane, so the number to "
                "read it against is the referee's free-space strip, not the ground-imaged "
                "one.",
        "n_turns": N_TURNS, "d_out": d_out, "d_in": d_in,
        "d_in_definition": "inner edge of the innermost complete turn (the inner extension "
                           "to the via sits inside the hole and is not a turn)",
        "L": float(sg.mohan_wheeler(N_TURNS, d_out, d_in, WIDTH, SPACING))}
    rec["mohan_wheeler"]["over_referee_no_ground_minus_1"] = \
        rec["mohan_wheeler"]["L"] / rec["strip_no_ground"] - 1.0
    rec["mohan_wheeler"]["over_referee_deembedded_minus_1"] = \
        rec["mohan_wheeler"]["L"] / rec["deembedded"] - 1.0
    ag = cv.area_gate(sg, THETA0, lead=LEAD)
    rec["area_gate"] = {"strip_polygon_area": ag["strip_polygon_area"],
                        "area_exact": ag["area_exact"], "greenhouse": ag["greenhouse"]}
    grad = []
    for k in range(3):
        def f(v: float, k: int = k) -> float:
            t = list(THETA0)
            t[k] = v
            return referee_deembedded(sg, cv, t)
        grad.append(float(fd4(f, THETA0[k], FD_STEP_REL * THETA0[k])))
    rec["gradient_fd4"] = grad
    rec["gradient_fd4_params"] = list(PARAMS)
    return rec


def fd4(f: Callable[[float], float], x0: float, d: float) -> float:
    return (-f(x0 + 2 * d) + 8 * f(x0 + d) - 8 * f(x0 - d) + f(x0 - 2 * d)) / (12 * d)


# ---------------------------------------------------------------------------
# 2. the FDFD side

def stack(eps: bool, t_si: float = T_SI, base_dz: float = BASE_DZ,
          metal_cells: int = METAL_CELLS):
    from rfx.fdfd import spiral as sm
    return sm.SmallStack(t_si=t_si, t_ox_low=T_OX_LOW, t_m1=T_M1, t_via=T_VIA, t_m2=T_M2,
                         t_ox_high=T_OX_HIGH, t_air=T_AIR, base_dz=base_dz,
                         metal_cells=metal_cells,
                         eps_si=EPS_SI if eps else 1.0, eps_ox=EPS_OX if eps else 1.0)


def build_level(div: float = 1.0, eps: bool = True, pad: int = PAD_CELLS,
                pad_z: int = PAD_CELLS_Z, t_si: float = T_SI, base_dz: float = BASE_DZ,
                metal_cells: int = METAL_CELLS, margin: float = MARGIN):
    """The static model at ``base_dx = WIDTH / div`` (see the module doc on
    why ``div`` barely moves the grid at this pitch)."""
    from rfx.fdfd import spiral as sm
    spec = sm.SpiralSpec(n_turns=N_TURNS, r_out=R_OUT, spacing=SPACING, width=WIDTH,
                         lead=LEAD, base_dx=WIDTH / div, margin=margin,
                         stack=stack(eps, t_si, base_dz, metal_cells), port_gap_cells=1,
                         pad_cells=pad, pad_cells_z=pad_z, pad_ratio=PAD_RATIO)
    return sm.build_spiral(spec)


# the grid the TEST runs live: the same RFIC geometry and the same code path
# (build -> three fixtures -> open/short de-embedding -> metrics) on a
# deliberately DEGRADED grid -- no wall padding, one cell per metal slab, one
# silicon cell -- so that one solve costs 13 s instead of 220 s. It is NOT a
# physics level and no accuracy claim is made on it: N = 26818 against the
# study's 85785, and its closed PEC box alone puts L_diff 1746.8 pH against
# the padded W/1 value. It exists so the test exercises the pipeline live
# rather than only reading the JSON.
CHEAP = {"pad": 0, "pad_z": 0, "metal_cells": 1, "base_dz": 2.0 * BASE_DZ,
         "margin": 0.5 * WIDTH}


def build_cheap(eps: bool = True):
    """The reduced-grid model of :data:`CHEAP` (see its comment)."""
    return build_level(1.0, eps=eps, **CHEAP)


def model_record(model) -> dict[str, Any]:
    z = np.asarray(model.z)
    return {"shape": list(model.shape), "n_unknowns": int(model.n_unknowns),
            "base_dx": float(model.spec.dx),
            "dx_min": float(np.diff(model.x_nom).min()),
            "dx_max": float(np.diff(model.x_nom).max()),
            "cells_across_width": int(round(WIDTH / float(np.diff(model.x_nom).min()))),
            "wall_x": [float(model.x_nom[0]), float(model.x_nom[-1])],
            "wall_y": [float(model.y_nom[0]), float(model.y_nom[-1])],
            "lid_z": float(z[-1]), "n_z_lines": int(len(z)),
            "dz": np.diff(z).tolist(), "t_si": float(model.spec.stack.t_si),
            "pad_cells": int(model.spec.pad_cells), "pad_cells_z": int(model.spec.pad_cells_z),
            "port_gap_m": float(z[model.spec.port_gap_cells] - z[0]),
            "build_seconds": float(model.build_seconds)}


def s_stats(s) -> dict[str, float]:
    s = np.asarray(s)
    return {"reciprocity": float(abs(s[0, 1] - s[1, 0])),
            "max_singular_value": float(np.linalg.svd(s, compute_uv=False).max())}


def solve_split(model, freq: float, theta=None, **kw) -> dict[str, Any]:
    """One de-embedded solve done as THREE separate single-fixture solves
    with the LU factor cache emptied between them, so that (a) the
    per-fixture factorisation+solve wall time is measured separately (gate
    R5) and (b) at most one LU factor is ever live. The de-embedding is then
    the same :mod:`rfx.fdfd.deembed` call ``solve_spiral`` makes, on the same
    three S-matrices, so the result is bit-identical to a single
    ``solve_spiral`` call at the same theta (checked in
    ``tests/unit/fdfd/test_fdfd_rfic_spiral.py``)."""
    import jax
    import jax.numpy as jnp

    from rfx.fdfd import deembed as de
    from rfx.fdfd import spiral as sm
    from rfx.fdfd.linear_solve import clear_factor_cache

    s: dict[str, Any] = {}
    secs: dict[str, float] = {}
    for name, field in (("dut", "s_raw"), ("open", "s_open"), ("short", "s_short")):
        t0 = time.time()
        r = sm.solve_spiral(model, freq, theta=theta, fixtures=(name,), **kw)
        v = getattr(r, field)
        jax.block_until_ready(v)
        secs[name] = time.time() - t0
        s[name] = np.asarray(v)
        clear_factor_cache()
        gc.collect()
    z0 = model.spec.z0
    f1 = jnp.asarray([freq], dtype=jnp.float64)
    z_dut = de.open_short_deembed(jnp.asarray(s["dut"])[:, :, None],
                                  jnp.asarray(s["open"])[:, :, None],
                                  jnp.asarray(s["short"])[:, :, None], z0)
    m = de.inductor_metrics(z_dut, f1)
    s_dut = np.asarray(de.z_to_s(z_dut, z0)[:, :, 0])
    z_raw = de.s_to_z(jnp.asarray(s["dut"])[:, :, None], z0)
    rec: dict[str, Any] = {
        "freq": float(freq), "theta": [float(v) for v in (theta if theta is not None
                                                          else model.theta_nominal)],
        "L_diff": float(m["L_diff"][0]), "Q_diff": float(m["Q_diff"][0]),
        "L_se": float(m["L_y11"][0]), "Q_se": float(m["Q_y11"][0]),
        "L_z11_open": float(m["L_se"][0]), "L_raw": float(de.l_diff(z_raw, f1)[0]),
        "seconds_per_fixture": secs, "seconds": float(sum(secs.values())),
        "z_dut": [[float(np.real(z_dut[i, j, 0])), float(np.imag(z_dut[i, j, 0]))]
                  for i in range(2) for j in range(2)],
        "s": {k: [[float(np.real(v[i, j])), float(np.imag(v[i, j]))]
                  for i in range(2) for j in range(2)] for k, v in
              list(s.items()) + [("deembedded", s_dut)]},
        "s_stats": {k: s_stats(v) for k, v in list(s.items()) + [("deembedded", s_dut)]},
        "mem": mem_note("solve_split"),
    }
    rec["reciprocity_worst_solved"] = max(rec["s_stats"][k]["reciprocity"]
                                          for k in ("dut", "open", "short"))
    rec["passivity_worst_solved"] = max(rec["s_stats"][k]["max_singular_value"]
                                        for k in ("dut", "open", "short"))
    spend()
    return rec


def metrics_fn(model, freq: float, theta0: Sequence[float] = THETA0, **kw) -> Callable:
    """``u -> (L_diff, Q_diff)`` on the SCALED parameters ``u = theta/theta0``
    (traced, jit-able)."""
    import jax.numpy as jnp

    from rfx.fdfd import spiral as sm
    s = jnp.asarray(np.asarray(theta0, dtype=np.float64))

    def raw(u):
        r = sm.solve_spiral(model, freq, theta=jnp.asarray(u) * s, **kw)
        return jnp.real(r.L_diff), jnp.real(r.Q_diff)

    return raw


def grad_block(model, freq: float, which: str, theta0: Sequence[float] = THETA0,
               **kw) -> dict[str, Any]:
    """``jax.value_and_grad`` of L_diff (``which="L"``) or Q_diff
    (``which="Q"``) w.r.t. theta at ``theta0``, one forward solve plus one
    reverse pass. The two are separate solve units on purpose: with the LU
    factor cache held at one entry (see the module doc) a reverse pass
    re-factorises the fixtures whose factor was evicted, and doing both
    pullbacks in one process would put nine factorisations in one chunk."""
    import jax
    import jax.numpy as jnp
    raw = metrics_fn(model, freq, theta0, **kw)
    idx = 0 if which == "L" else 1
    fn = jax.jit(jax.value_and_grad(lambda u: raw(u)[idx]))
    t0 = time.time()
    val, g = fn(jnp.ones(3, dtype=jnp.float64))
    jax.block_until_ready(g)
    sec = time.time() - t0
    s = np.asarray(theta0, dtype=np.float64)
    g_scaled = np.asarray(g, dtype=np.float64)
    spend()
    return {"which": which, "value": float(val), "grad_scaled": g_scaled.tolist(),
            "grad": (g_scaled / s).tolist(), "params": list(PARAMS),
            "theta0": [float(v) for v in theta0], "seconds": sec,
            "mem": mem_note(f"grad_{which}")}


def forward_point(model, freq: float, theta, **kw) -> dict[str, Any]:
    """One forward three-fixture solve of (L_diff, Q_diff) at ``theta``: the
    FD stencil's evaluation, and the cheapest unit that exists."""
    import jax
    import jax.numpy as jnp

    from rfx.fdfd import spiral as sm
    t0 = time.time()
    r = sm.solve_spiral(model, freq, theta=jnp.asarray(np.asarray(theta, dtype=np.float64)), **kw)
    jax.block_until_ready(r.L_diff)
    sec = time.time() - t0
    spend()
    return {"theta": [float(v) for v in np.asarray(theta)], "L_diff": float(r.L_diff),
            "Q_diff": float(r.Q_diff), "L_se": float(r.L_se), "Q_se": float(r.Q_se),
            "seconds": sec, "mem": mem_note("forward")}


# ---------------------------------------------------------------------------
# 3. part A: absolute L at 100 MHz against the referee

def part_a(study: dict[str, Any], dump: Callable[[], None]) -> None:
    ref = study["referee"]["deembedded"]
    a = study.setdefault("A", {"what":
        "absolute L at 100 MHz under D2's validated protocol (uniform-current volumetric "
        "metal sigma = 3e6, vacuum dielectrics, pad_cells = 4, area-exact corners) against "
        "the PRIMARY referee L(strip) - L(bridge); no FDFD-side correction applied",
        "freq": FREQ_A, "sigma_volumetric": SIGMA_VOL, "levels": {}})
    for div in (DIVS if want_a("levels") else ()):
        key = f"{div:g}"
        if key in a["levels"]:
            continue
        check_budget()
        model = build_level(div, eps=False)
        rec = solve_split(model, FREQ_A, sigma_volumetric=SIGMA_VOL)
        rec["grid"] = model_record(model)
        rec["referee_deembedded"] = ref
        rec["gap"] = rec["L_diff"] / ref - 1.0
        rec["referee_strip_only"] = study["referee"]["strip"]
        rec["gap_strip_only"] = rec["L_diff"] / study["referee"]["strip"] - 1.0
        rec["d2_same_cells_across_width"] = D2_GAPS.get(
            f"W/{rec['grid']['cells_across_width']:g}")
        a["levels"][key] = rec
        print(f"  A W/{div:g}: N={rec['grid']['n_unknowns']} "
              f"{rec['grid']['cells_across_width']} cells across W  "
              f"L_diff={rec['L_diff'] * 1e12:.2f} pH vs referee {ref * 1e12:.2f} pH "
              f"= {100 * rec['gap']:+.2f} %  ({rec['seconds']:.0f} s)", flush=True)
        dump()
    # the SAME wall ladder as part B, but under part A's protocol and at
    # 100 MHz, so the wall bias on the R1 gap is a measurement and not an
    # inference from the 2.45 GHz number
    wall = a.setdefault("wall", {})
    for pad in (WALL_PADS[-1],) if want_a("wall") else ():
        key = str(pad)
        if key in wall:
            continue
        check_budget()
        model = build_level(1.0, eps=False, pad=pad)
        rec = solve_split(model, FREQ_A, sigma_volumetric=SIGMA_VOL)
        rec["grid"] = model_record(model)
        rec["gap"] = rec["L_diff"] / ref - 1.0
        wall[key] = rec
        print(f"  A wall pad={pad}: N={rec['grid']['n_unknowns']} "
              f"wall={rec['grid']['wall_x'][1] * 1e6:.0f} um "
              f"L_diff={rec['L_diff'] * 1e12:.2f} pH = {100 * rec['gap']:+.2f} % vs referee",
              flush=True)
        dump()

    if "w3_projection" not in a:
        model = build_level(DIV_W3, eps=False)
        g = model_record(model)
        fine = a["levels"].get("1", {}).get("grid")
        a["w3_projection"] = {
            "what": "the level the bar study called the 2 % point (3+ cells across W): built "
                    "but NOT solved -- N and the extrapolated cost only",
            "grid": g, "n_unknowns_ratio_to_W1": (g["n_unknowns"] / fine["n_unknowns"]
                                                  if fine else None)}
        dump()


# ---------------------------------------------------------------------------
# 4. part B: 2.45 GHz, Leontovich metal, lossy silicon

def leontovich_block(model) -> dict[str, Any]:
    from rfx.fdfd import spiral as sm
    d_sg = float(sm.skin_depth(FREQ_B, SIGMA_CU))
    d_cu = float(sm.skin_depth(FREQ_B, SIGMA_CU_PURE))
    return {
        "what": "skin depth over the THINNEST metal slab of the stack; the Leontovich sheet "
                "needs it well below 1 and it is not -- absolute Q at 2.45 GHz is therefore "
                "not a validated number here (see the module doc)",
        "freq": FREQ_B, "sigma_metal": SIGMA_CU, "skin_depth": d_sg,
        "t_min": float(min(T_M1, T_VIA, T_M2)), "t_m2": T_M2,
        "leontovich_validity": float(sm.leontovich_validity(model, FREQ_B, SIGMA_CU)),
        "delta_over_t_m2": d_sg / T_M2,
        "pure_copper": {"sigma": SIGMA_CU_PURE, "skin_depth": d_cu,
                        "delta_over_t_m2": d_cu / T_M2,
                        "note": "the task's expected ~0.45 is delta/t for PURE copper "
                                "(5.8e7 S/m) on the 3 um TopMetal2; the stack's 3.05e7 S/m "
                                "gives a 1.36x larger skin depth"},
    }


def scaling_block(dump: Callable[[], None], rec: dict[str, Any],
                  pads: Sequence[int] = SCALING_PADS, time_budget: float = 380.0
                  ) -> dict[str, Any]:
    """The cost law of THIS fixture family, measured, so that the cost of the
    level nobody can afford is an extrapolation of measurements instead of a
    guess.

    One SINGLE-FIXTURE solve (the DUT alone -- one factorisation, one pair of
    port excitations) per wall-padding count, which is the cleanest way to
    move N without changing the geometry, the stack or the vertical grid.
    Each point records N, the wall time and the physical-footprint INCREMENT
    across the solve (the factor's own cost; the process baseline is
    subtracted), with the LU factor cache emptied and ``gc.collect()`` forced
    between points. The block does not spend solve units -- every point is
    cheaper than the study's own levels -- but it stops when
    ``time_budget`` seconds are gone, so it cannot overrun a chunk."""
    import jax

    from rfx.fdfd import spiral as sm
    from rfx.fdfd.linear_solve import clear_factor_cache

    rec.setdefault("what",
        "single-fixture (DUT only) solve time and factor memory against N on one geometry, "
        "N moved by the wall padding alone; the fit is the basis of the W/3 cost projection")
    pts = rec.setdefault("points", {})
    t_block = time.time()
    for pad in pads:
        key = str(pad)
        if key in pts or time.time() - t_block > time_budget:
            continue
        model = build_level(1.0, pad=pad, pad_z=min(pad, PAD_CELLS_Z))
        clear_factor_cache()
        gc.collect()
        m0 = proc_memory()
        t0 = time.time()
        r = sm.solve_spiral(model, FREQ_B, sigma_metal=SIGMA_CU, sigma_si=SIGMA_SI,
                            fixtures=("dut",))
        jax.block_until_ready(r.s_raw)
        sec = time.time() - t0
        m1 = proc_memory()
        pts[key] = {"pad_cells": pad, "n_unknowns": int(model.n_unknowns),
                    "shape": list(model.shape), "seconds": sec,
                    "footprint_before_gb": m0["footprint_gb"],
                    "footprint_after_gb": m1["footprint_gb"],
                    "footprint_increment_gb": m1["footprint_gb"] - m0["footprint_gb"],
                    "rss_after_gb": m1["rss_gb"]}
        print(f"    scaling pad={pad}: N={model.n_unknowns} {sec:.1f} s/fixture "
              f"+{pts[key]['footprint_increment_gb']:.2f} GB", flush=True)
        clear_factor_cache()
        gc.collect()
        dump()
    if len(pts) >= 3:
        order = sorted(pts, key=lambda k: pts[k]["n_unknowns"])
        n = np.array([pts[k]["n_unknowns"] for k in order], dtype=float)
        s = np.array([pts[k]["seconds"] for k in order], dtype=float)
        inc = np.array([pts[k]["footprint_increment_gb"] for k in order], dtype=float)
        after = np.array([pts[k]["footprint_after_gb"] for k in order], dtype=float)
        p_t = float(np.polyfit(np.log(n), np.log(s), 1)[0])
        # MEMORY: the per-solve footprint INCREMENT is only meaningful in a
        # process that starts clean -- run after a reverse pass it can even be
        # negative, because the previous pass's pages are released during this
        # solve. The robust quantity is the ABSOLUTE footprint after the solve
        # (the process has only ever solved smaller systems, so it is that
        # system's peak), and the fit is on that; the increments are kept and
        # flagged.
        rec["memory_increments_usable"] = bool(np.all(inc > 0))
        p_m = float(np.polyfit(np.log(n), np.log(after), 1)[0])
        rec["exponent_time"] = p_t
        rec["exponent_memory"] = p_m
        rec["memory_basis"] = "footprint_after_gb"
        rec["fit_note"] = ("least squares on log N over N = %d..%d (%d points): seconds ~ "
                           "N^%.2f, peak process footprint ~ N^%.2f%s" % (
                               int(n.min()), int(n.max()), len(pts), p_t, p_m,
                               "" if rec["memory_increments_usable"] else
                               " (the per-solve increments are NOT usable in this run: "
                               "at least one is negative, see memory_increments_usable)"))
        rec["footprint_after_gb"] = after.tolist()
        rec["seconds"] = s.tolist()
        rec["n_unknowns"] = n.tolist()
    return rec


def part_b(study: dict[str, Any], dump: Callable[[], None]) -> None:
    b = study.setdefault("B", {"what":
        "2.45 GHz with a Leontovich metal sheet (sigma 3.05e7) and lossy silicon "
        "(sigma 2 S/m) and the real permittivities (eps_si 11.9, eps_ox 4.1)",
        "freq": FREQ_B, "sigma_metal": SIGMA_CU, "sigma_si": SIGMA_SI,
        "eps_si": EPS_SI, "eps_ox": EPS_OX, "levels": {}, "sweep": {},
        "t_si_sensitivity": {}, "wall": {}, "gradients": {}, "fd": {}})
    kw = dict(sigma_metal=SIGMA_CU, sigma_si=SIGMA_SI)

    if want("leontovich") and "leontovich" not in b:
        b["leontovich"] = leontovich_block(build_level(1.0))
        v = b["leontovich"]
        print(f"  B leontovich: delta={v['skin_depth'] * 1e6:.3f} um  "
              f"delta/t_min={v['leontovich_validity']:.3f}  "
              f"delta/t_M2={v['delta_over_t_m2']:.3f}  "
              f"(pure Cu delta/t_M2={v['pure_copper']['delta_over_t_m2']:.3f})", flush=True)
        dump()

    for div in (DIVS if want("levels") else ()):
        key = f"{div:g}"
        if key in b["levels"]:
            continue
        check_budget()
        model = build_level(div)
        rec = solve_split(model, FREQ_B, **kw)
        rec["grid"] = model_record(model)
        b["levels"][key] = rec
        print(f"  B W/{div:g}: N={rec['grid']['n_unknowns']} L_diff={rec['L_diff'] * 1e12:.1f} pH "
              f"Q_diff={rec['Q_diff']:.3f} L_se={rec['L_se'] * 1e12:.1f} pH "
              f"Q_se={rec['Q_se']:.3f}  recip={rec['reciprocity_worst_solved']:.2e} "
              f"passivity={rec['passivity_worst_solved']:.12f}  ({rec['seconds']:.0f} s)",
              flush=True)
        dump()

    # substrate thickness sensitivity at the coarse level (two values; the
    # chosen one is the W/1 solve above)
    if want("t_si") and "1" in b["levels"] and "alt" not in b["t_si_sensitivity"]:
        check_budget()
        model = build_level(1.0, t_si=T_SI_ALT)
        rec = solve_split(model, FREQ_B, **kw)
        rec["grid"] = model_record(model)
        base = b["levels"]["1"]
        ts = b["t_si_sensitivity"]
        ts.update({
            "what": "L and Q against the substrate thickness at base_dx = W, 2.45 GHz: the "
                    "fixture's PEC ground sits under the silicon, so the thickness is the "
                    "ground-image distance and it is bought in whole 60 um cells",
            "chosen": {"t_si": T_SI, "n_si_cells": int(round(T_SI / BASE_DZ)),
                       "L_diff": base["L_diff"], "Q_diff": base["Q_diff"],
                       "L_se": base["L_se"], "Q_se": base["Q_se"],
                       "n_unknowns": base["grid"]["n_unknowns"]},
            "alt": {"t_si": T_SI_ALT, "n_si_cells": int(round(T_SI_ALT / BASE_DZ)),
                    "L_diff": rec["L_diff"], "Q_diff": rec["Q_diff"],
                    "L_se": rec["L_se"], "Q_se": rec["Q_se"],
                    "n_unknowns": rec["grid"]["n_unknowns"], "solve": rec},
            "dL_rel": rec["L_diff"] / base["L_diff"] - 1.0,
            "dQ_rel": rec["Q_diff"] / base["Q_diff"] - 1.0,
            "referee_curve": study["referee"]["ground_image_curve"],
            "referee_dL_rel_same_pair": (
                study["referee"]["ground_image_curve"][f"{T_SI_ALT * 1e6:g}"]["deembedded"]
                / study["referee"]["ground_image_curve"][f"{T_SI * 1e6:g}"]["deembedded"] - 1.0),
            "referee_dL_rel_chosen_to_real": (
                study["referee"]["ground_image_curve"][f"{T_SI_REAL * 1e6:g}"]["deembedded"]
                / study["referee"]["ground_image_curve"][f"{T_SI * 1e6:g}"]["deembedded"] - 1.0),
        })
        print(f"  B t_si {T_SI * 1e6:g}->{T_SI_ALT * 1e6:g} um: dL={100 * ts['dL_rel']:+.2f} % "
              f"dQ={100 * ts['dQ_rel']:+.2f} % (referee dL={100 * ts['referee_dL_rel_same_pair']:+.2f} %)",
              flush=True)
        dump()

    # frequency sweep at the coarse level
    for f in (SWEEP_FREQS if want("sweep") else ()):
        key = f"{f / 1e9:g}"
        if key in b["sweep"]:
            continue
        if abs(f - FREQ_B) < 1e-6 and "1" in b["levels"]:
            src = b["levels"]["1"]
            b["sweep"][key] = {k: src[k] for k in
                               ("freq", "L_diff", "Q_diff", "L_se", "Q_se", "L_raw",
                                "reciprocity_worst_solved", "passivity_worst_solved",
                                "seconds")}
            b["sweep"][key]["from"] = "levels.1 (same solve)"
            dump()
            continue
        check_budget()
        model = build_level(1.0)
        rec = solve_split(model, f, **kw)
        rec["grid"] = model_record(model)
        b["sweep"][key] = rec
        print(f"  B sweep {f / 1e9:g} GHz: L_diff={rec['L_diff'] * 1e12:.1f} pH "
              f"Q_diff={rec['Q_diff']:.3f}  ({rec['seconds']:.0f} s)", flush=True)
        dump()

    # side-wall independence
    for pad in (WALL_PADS if want("wall") else ()):
        key = str(pad)
        if key in b["wall"]:
            continue
        if pad == PAD_CELLS and "1" in b["levels"]:
            src = b["levels"]["1"]
            b["wall"][key] = {"pad_cells": pad, "L_diff": src["L_diff"],
                              "Q_diff": src["Q_diff"], "n_unknowns": src["grid"]["n_unknowns"],
                              "wall_x": src["grid"]["wall_x"], "seconds": src["seconds"],
                              "from": "levels.1 (same solve)"}
            dump()
            continue
        check_budget()
        model = build_level(1.0, pad=pad)
        rec = solve_split(model, FREQ_B, **kw)
        rec["grid"] = model_record(model)
        b["wall"][key] = {"pad_cells": pad, "L_diff": rec["L_diff"], "Q_diff": rec["Q_diff"],
                          "n_unknowns": rec["grid"]["n_unknowns"],
                          "wall_x": rec["grid"]["wall_x"], "seconds": rec["seconds"],
                          "solve": rec}
        print(f"  B wall pad={pad}: N={rec['grid']['n_unknowns']} "
              f"wall={rec['grid']['wall_x'][1] * 1e6:.0f} um "
              f"L_diff={rec['L_diff'] * 1e12:.1f} pH Q={rec['Q_diff']:.3f}", flush=True)
        dump()
    if all(str(p) in b["wall"] for p in WALL_PADS):
        lo = b["wall"][str(WALL_PADS[0])]
        hi = b["wall"][str(WALL_PADS[-1])]
        b["wall"]["rel_change_L"] = hi["L_diff"] / lo["L_diff"] - 1.0
        b["wall"]["rel_change_Q"] = hi["Q_diff"] / lo["Q_diff"] - 1.0
        # the wall bias is a 1/distance effect (the straight-bar study measured
        # the same ladder): fit L = L_inf - c / d on the three pads and quote
        # the extrapolated wall-free value and what it does to the pad-4 number
        d = np.array([b["wall"][str(p)]["wall_x"][1] for p in WALL_PADS], dtype=float)
        lv_ = np.array([b["wall"][str(p)]["L_diff"] for p in WALL_PADS], dtype=float)
        qv_ = np.array([b["wall"][str(p)]["Q_diff"] for p in WALL_PADS], dtype=float)
        a_l, b_l = np.polyfit(1.0 / d, lv_, 1)
        a_q, b_q = np.polyfit(1.0 / d, qv_, 1)
        b["wall"]["ladder"] = {
            "what": "least squares of L (and Q) against 1/wall_distance over the three pads: "
                    "the intercept is the wall-free limit",
            "pads": list(WALL_PADS), "wall_distance": d.tolist(),
            "L_diff": lv_.tolist(), "Q_diff": qv_.tolist(),
            "L_inf": float(b_l), "slope_L": float(a_l),
            "Q_inf": float(b_q), "slope_Q": float(a_q),
            "L_inf_over_pad0_minus_1": float(b_l / lv_[0] - 1.0),
            "L_inf_over_padlast_minus_1": float(b_l / lv_[-1] - 1.0),
            "Q_inf_over_pad0_minus_1": float(b_q / qv_[0] - 1.0),
            "residual_L": float(np.max(np.abs(lv_ - (b_l + a_l / d))) / lv_[0])}
        lid = b["levels"]["1"]["grid"]["lid_z"] if "1" in b["levels"] else float("nan")
        b["wall"]["note"] = (
            "pad_cells_z is pinned at %d in both pads, so this gate moves the four SIDE walls "
            "only (%.0f -> %.0f um from the axis); the PEC lid stays at z = %.0f um, i.e. "
            "%.0f um above the top of TopMetal2. D2's wall gate moved the lid with the walls "
            "and measured the two together as +0.114 %%." % (
                PAD_CELLS_Z, lo["wall_x"][1] * 1e6, hi["wall_x"][1] * 1e6, lid * 1e6,
                (lid - (T_SI + T_OX_LOW + T_M1 + T_VIA + T_M2)) * 1e6))
        dump()

    # gradients and their FD4 check
    for which in (("L", "Q") if want("grad") else ()):
        if which in b["gradients"]:
            continue
        check_budget()
        model = build_level(1.0)
        rec = grad_block(model, FREQ_B, which, **kw)
        b["gradients"][which] = rec
        print(f"  B grad {which}: {rec['grad']}  ({rec['seconds']:.0f} s)", flush=True)
        dump()

    if want("scaling") and len(b.get("scaling", {}).get("points", {})) < len(SCALING_PADS):
        scaling_block(dump, b.setdefault("scaling", {}))
        dump()

    # FD4: 4 forward solves per parameter, each recorded so a killed chunk keeps it
    for k, name in enumerate(PARAMS):
        if name not in ("r_out", "width") or not want("fd"):
            continue
        st = b["fd"].setdefault(name, {"param": name, "index": k,
                                       "step": FD_STEP_REL * THETA0[k], "points": {}})
        for mult in (-2, -1, 1, 2):
            pk = str(mult)
            if pk in st["points"]:
                continue
            check_budget()
            model = build_level(1.0)
            theta = list(THETA0)
            theta[k] = THETA0[k] + mult * st["step"]
            rec = forward_point(model, FREQ_B, theta, **kw)
            st["points"][pk] = rec
            print(f"  B fd {name} {mult:+d}: L={rec['L_diff'] * 1e12:.3f} pH "
                  f"Q={rec['Q_diff']:.5f}  ({rec['seconds']:.0f} s)", flush=True)
            dump()
        if len(st["points"]) == 4:
            d = st["step"]
            for field in ("L_diff", "Q_diff"):
                p = {int(m): st["points"][m][field] for m in st["points"]}
                st[f"fd4_d{field}"] = (-p[2] + 8 * p[1] - 8 * p[-1] + p[-2]) / (12 * d)
            dump()


# ---------------------------------------------------------------------------
# 5. part C: the design loop

def loss_of(l_value, q_value, l_target: float, q_ref: float, lam: float):
    return -q_value / q_ref + lam * ((l_value - l_target) / l_target) ** 2


def objective_fn(model, l_target: float, q_ref: float, lam: float, freq: float, **kw):
    import jax
    raw = metrics_fn(model, freq, THETA0, **kw)

    def loss(u):
        l_value, q_value = raw(u)
        return loss_of(l_value, q_value, l_target, q_ref, lam), (l_value, q_value)

    return jax.jit(jax.value_and_grad(loss, has_aux=True))


def bounds_scaled() -> list[list[float]]:
    return [[lo / THETA0[k], hi / THETA0[k]] for k, (lo, hi) in enumerate(BOUNDS)]


def feasibility_margin(theta) -> float:
    """``a_in - width/2`` (m): the binding topological constraint, for
    ``n_turns = 3`` it is ``r_out - width - n (width + spacing)``
    (``spiral.check_feasible`` raises when it is <= 0)."""
    r_out, spacing, width = (float(v) for v in theta)
    return r_out - width - N_TURNS * (width + spacing)


def run_loop(model, c: dict[str, Any], dump: Callable[[], None], **kw) -> None:
    """L-BFGS-B on ``u = theta/theta0`` with ``jac=True`` from
    ``jax.value_and_grad``, RESUMABLE BY REPLAY: every evaluation is logged
    with its ``u``, loss and gradient, and a resumed run re-runs ``minimize``
    from scratch serving ``fun`` from the log (checked bitwise, order
    included) until the log is exhausted, then continues with real solves
    until the solve budget stops it. The pattern, the bitwise replay check
    and the reason for it (L-BFGS-B holds unpicklable state; one solve unit
    per process is a memory requirement) are
    ``validation/fdfd/spiral_design.py``'s."""
    import jax
    import jax.numpy as jnp
    from scipy.optimize import minimize

    from rfx.fdfd import spiral as sm
    from rfx.fdfd.linear_solve import clear_factor_cache

    vg = objective_fn(model, c["L_target"], c["Q_ref"], c["lam"], **kw)
    s = np.asarray(THETA0, dtype=np.float64)
    evals: list[dict[str, Any]] = c["evals"]
    queue = [dict(e) for e in evals]
    state = {"replayed": 0, "mismatches": 0, "new": 0}

    def fun(u: np.ndarray):
        if queue:
            e = queue.pop(0)
            if [float(v) for v in u] != [float(v) for v in e["u"]]:
                state["mismatches"] += 1
                c["n_replay_mismatches"] = state["mismatches"]
                raise RuntimeError(f"replay diverged at evaluation {e['i']}: L-BFGS-B asked "
                                   f"for {list(np.asarray(u))} against the logged {e['u']}")
            state["replayed"] += 1
            return float(e["loss"]), np.asarray(e["grad"], dtype=np.float64)
        check_budget()
        t0 = time.time()
        theta = np.asarray(u, dtype=np.float64) * s
        sm.check_feasible(model, theta)
        (loss, (l_value, q_value)), g = vg(jnp.asarray(u, dtype=jnp.float64))
        jax.block_until_ready(g)
        g = np.asarray(g, dtype=np.float64)
        evals.append({"i": len(evals), "u": [float(v) for v in u], "theta": theta.tolist(),
                      "L": float(l_value), "Q": float(q_value), "loss": float(loss),
                      "grad": g.tolist(), "grad_inf": float(np.max(np.abs(g))),
                      "L_rel_error": float(l_value) / c["L_target"] - 1.0,
                      "feasibility_margin": feasibility_margin(theta),
                      "seconds": time.time() - t0, "accepted": False,
                      "mem": mem_note(f"loop:eval{len(evals)}")})
        clear_factor_cache()
        gc.collect()
        state["new"] += 1
        spend()
        e = evals[-1]
        print(f"    eval {e['i']:3d} theta=({theta[0] * 1e6:7.3f}, {theta[1] * 1e6:6.3f}, "
              f"{theta[2] * 1e6:6.3f}) um  L={e['L'] * 1e12:8.1f} pH ({100 * e['L_rel_error']:+6.2f} %) "
              f"Q={e['Q']:7.4f} loss={e['loss']:+.6f} |g|inf={e['grad_inf']:.4g} "
              f"{e['seconds']:.0f} s", flush=True)
        dump()
        return float(loss), g

    x0 = np.ones(3, dtype=np.float64)
    accepted: list[np.ndarray] = [x0.copy()]
    if "gtol" not in c:
        f0, g0 = fun(x0)
        c["loss_at_nominal"] = float(f0)
        c["grad_inf_at_nominal"] = float(np.max(np.abs(g0)))
        c["gtol"] = GTOL_FACTOR * c["grad_inf_at_nominal"]
        dump()
        queue.clear()
        queue.extend(dict(e) for e in evals)
        state["replayed"] = 0

    try:
        res = minimize(fun, x0, jac=True, method="L-BFGS-B", bounds=bounds_scaled(),
                       callback=lambda xk, *a, **k: accepted.append(
                           np.asarray(xk, dtype=np.float64).copy()),
                       options={"maxiter": MAXITER, "gtol": c["gtol"], "ftol": 1e-12,
                                "maxcor": 10})
    except BudgetStop:
        c.update(complete=False, n_evals_logged=len(evals),
                 n_new_solves_this_chunk=state["new"],
                 n_evals_replayed=state["replayed"],
                 n_replay_mismatches=state["mismatches"])
        print(f"    budget reached after {len(evals)} logged evaluations "
              f"({state['new']} new this chunk); rerun to continue by replay", flush=True)
        raise
    for xa in accepted:
        hit = [e for e in evals if np.allclose(np.asarray(e["u"]), xa, rtol=0, atol=1e-14)]
        if hit:
            hit[0]["accepted"] = True
    final = [e for e in evals if e["accepted"]][-1]
    c.update(complete=True, n_evals_replayed=state["replayed"],
             n_new_solves_this_chunk=state["new"],
             n_replay_mismatches=state["mismatches"],
             n_objective_evaluations=len(evals),
             n_forward_solves=len(evals), n_adjoint_passes=len(evals),
             n_accepted_iterates=int(sum(e["accepted"] for e in evals)),
             n_scipy_iterations=int(res.nit), n_scipy_funcalls=int(res.nfev),
             status=int(res.status), message=str(res.message), success=bool(res.success),
             final=final, final_u=final["u"], final_theta=final["theta"],
             final_L=final["L"], final_Q=final["Q"], final_loss=final["loss"],
             final_L_rel_error=final["L_rel_error"], final_grad=final["grad"])
    print(f"  C L-BFGS-B: {res.nit} iterations, {len(evals)} evaluations ({res.message})",
          flush=True)
    dump()


def part_c(study: dict[str, Any], dump: Callable[[], None]) -> None:
    b = study.get("B", {})
    if "1" not in b.get("levels", {}) or "L" not in b.get("gradients", {}) \
            or "Q" not in b.get("gradients", {}):
        print("  C: needs B levels.1 and both gradients first", flush=True)
        return
    kw = dict(sigma_metal=SIGMA_CU, sigma_si=SIGMA_SI)
    base = b["levels"]["1"]
    gl = np.asarray(b["gradients"]["L"]["grad_scaled"], dtype=np.float64)
    gq = np.asarray(b["gradients"]["Q"]["grad_scaled"], dtype=np.float64)
    n_l = float(np.linalg.norm(gl / L_TARGET))
    n_q = float(np.linalg.norm(gq / base["Q_diff"]))
    c = study.setdefault(_C_KEY, {})
    if "lam" not in c:
        lam_rule = n_q / (2.0 * E_TARGET * n_l)
        c.update({
            "what": "L-BFGS-B design at the coarse level: hit L_diff = 4.000 nH within 2 %% "
                    "while maximising Q_diff over (r_out, spacing, width) in a feasible box "
                    "around the paper's values. The 4.000 nH / 16.80 of the paper are a "
                    "TARGET and a context number, not a physics gate: the DUT is a square "
                    "spiral, the model is the biased coarse grid of part A, and the optimum "
                    "found here is the optimum OF THAT MODEL.",
            "freq": FREQ_B, "L_target": L_TARGET, "Q_ref": base["Q_diff"],
            "Q_paper_context": Q_PAPER,
            "bounds": [list(x) for x in BOUNDS], "bounds_scaled": bounds_scaled(),
            "theta_nominal": list(THETA0), "maxiter": MAXITER,
            "L_at_nominal": base["L_diff"], "Q_at_nominal": base["Q_diff"],
            "L_rel_error_at_nominal": base["L_diff"] / L_TARGET - 1.0,
            "feasibility_margin_nominal": feasibility_margin(THETA0),
            "lam_rule": "lam = |g_Q/Q_ref| / (2 e_target |g_L/L_target|) (scaled units): at "
                        "an interior stationary point -g_Q + 2 lam e g_L = 0, so the relative "
                        "L error is bounded by e_target. The two gradients are B's, measured "
                        "at the nominal theta on this very model -- no extra solve.",
            "e_target": E_TARGET, "norm_g_L_over_Lt": n_l, "norm_g_Q_over_Qref": n_q,
            "lam_from_nominal_rule": lam_rule,
            "lam": lam_rule if _LAM is None else float(_LAM),
            "lam_override": _LAM,
            "lam_override_rationale": None if _LAM is None else (
                "the nominal-gradient rule gives lam = %.4f, and a run at that weight "
                "(C_attempt_1 in this JSON) showed it does NOT enforce the constraint: the "
                "loop moved from 1.72 %% to 6.35 %% L error to gain 1.1 %% in Q and the loss "
                "went DOWN, so the Q-vs-L trade rate at the optimum is far steeper than the "
                "nominal gradient ratio predicts. Measured on that run's own log between its "
                "evaluations 5 and 7: d(Q/Q_ref)/d|e| = 0.0112/0.0463 = 0.242 in loss units, "
                "so the stationary error under a weight lam is |e| = 0.242/(2 lam) and "
                "lam = %.4g pins it at %.2f %%." % (lam_rule, float(_LAM),
                                                    100 * 0.242 / (2.0 * float(_LAM)))),
            "evals": []})
        dump()
    model = build_level(1.0)
    if not c.get("complete") and want_c("loop"):
        run_loop(model, c, dump, freq=FREQ_B, **kw)
    if not c.get("complete"):
        return

    # FD4 of the OBJECTIVE at the optimum, on the scaled variables.
    #
    # A bound-constrained optimum SITS ON BOUNDS -- Q_diff is monotone in
    # spacing and width over this box, so those two are expected to end at
    # their lower limits -- and a central stencil there would ask for
    # infeasible parameters. Each parameter therefore gets the 4th-order
    # stencil that FITS INSIDE THE BOX: central ``(-f2 + 8f1 - 8f-1 + f-2)/12h``
    # when both wings fit, otherwise the one-sided 4th-order formula
    # ``(-25f0 + 48f1 - 36f2 + 16f3 - 3f4)/12h`` (forward) or its mirror
    # (backward), which needs the value AT the optimum: that is solved once
    # and shared (``points["base"]``), and it doubles as a check of the
    # jitted ``value_and_grad`` forward against an eager forward solve at the
    # same theta. Same four new solves per parameter either way, same order.
    fdrec = c.setdefault("fd_at_optimum", {"step_rel": FD_STEP_REL, "points": {},
                                           "stencils": {}})
    u_star = np.asarray(c["final_u"], dtype=np.float64)
    bs = bounds_scaled()
    h = FD_STEP_REL
    tol = 1e-12

    def stencil_for(k: int) -> tuple[str, tuple[int, ...]]:
        lo, hi = bs[k]
        u = float(u_star[k])
        if u - 2 * h >= lo - tol and u + 2 * h <= hi + tol:
            return "central", (-2, -1, 1, 2)
        if u + 4 * h <= hi + tol:
            return "forward", (1, 2, 3, 4)
        if u - 4 * h >= lo - tol:
            return "backward", (-1, -2, -3, -4)
        return "none", ()

    def loss_at(u: np.ndarray, tag: str) -> dict[str, Any]:
        check_budget()
        rec = forward_point(model, FREQ_B, u * np.asarray(THETA0), **kw)
        rec["loss"] = float(loss_of(rec["L_diff"], rec["Q_diff"], c["L_target"],
                                    c["Q_ref"], c["lam"]))
        rec["u"] = [float(v) for v in u]
        print(f"    C fd {tag}: loss={rec['loss']:+.8f} L={rec['L_diff'] * 1e12:.2f} pH "
              f"Q={rec['Q_diff']:.4f}", flush=True)
        return rec

    todo = [(k, name) for k, name in enumerate(PARAMS)
            if want_c("fd_opt") and want_fd_param(name)]
    for k, name in todo:
        fdrec["stencils"][name] = stencil_for(k)[0]
    if any(fdrec["stencils"].get(n) in ("forward", "backward") for _, n in todo) \
            and "base" not in fdrec["points"]:
        fdrec["points"]["base"] = loss_at(u_star, "base (at the optimum)")
        fdrec["points"]["base"]["loop_logged_loss"] = c["final_loss"]
        fdrec["points"]["base"]["loss_minus_loop_logged"] = (
            fdrec["points"]["base"]["loss"] - c["final_loss"])
        dump()
    for k, name in todo:
        kind, mults = stencil_for(k)
        if kind == "none":
            fdrec["points"][f"{name}:skipped"] = {
                "skipped": "no 4th-order stencil fits between the bounds at this optimum",
                "u": float(u_star[k]), "bounds": list(bs[k]), "step": h}
            dump()
            continue
        for mult in mults:
            pk = f"{name}:{mult}"
            if pk in fdrec["points"]:
                continue
            u = u_star.copy()
            u[k] = u_star[k] + mult * h
            fdrec["points"][pk] = loss_at(u, f"{name} {mult:+d}")
            dump()
        vals = {m: fdrec["points"].get(f"{name}:{m}") for m in mults}
        if not all(v and "loss" in v for v in vals.values()):
            continue
        f = {m: vals[m]["loss"] for m in mults}
        if kind == "central":
            g = (-f[2] + 8 * f[1] - 8 * f[-1] + f[-2]) / (12 * h)
        else:
            f0 = fdrec["points"]["base"]["loss"]
            if kind == "forward":
                g = (-25 * f0 + 48 * f[1] - 36 * f[2] + 16 * f[3] - 3 * f[4]) / (12 * h)
            else:
                g = (25 * f0 - 48 * f[-1] + 36 * f[-2] - 16 * f[-3] + 3 * f[-4]) / (12 * h)
        fdrec.setdefault("fd4", {})[name] = float(g)
        dump()
    if "fd4" in fdrec and len(fdrec["fd4"]) == 3:
        g = np.asarray(c["final_grad"], dtype=np.float64)
        f4 = np.asarray([fdrec["fd4"][n] for n in PARAMS], dtype=np.float64)
        fdrec["grad_ad"] = g.tolist()
        fdrec["params"] = list(PARAMS)
        fdrec["rel"] = [abs(a - b) / abs(b) if b != 0 else float("nan")
                        for a, b in zip(g, f4)]
        fdrec["worst_rel"] = float(np.nanmax(fdrec["rel"]))
        fdrec["projected_gradient_note"] = (
            "the AD gradient is the FULL gradient, not the projected one; at a parameter "
            "sitting on a bound it points OUT of the box and L-BFGS-B is stationary "
            "anyway. The FD4 comparison is of the full derivative, which is what AD "
            "computes.")
        dump()

    # the W/2 re-solve of the optimum (the honest statement of the bias's effect)
    if "w2_resolve" not in c and want_c("w2"):
        check_budget()
        m2 = build_level(2.0)
        rec = solve_split(m2, FREQ_B, theta=np.asarray(c["final_theta"]), **kw)
        rec["grid"] = model_record(m2)
        c["w2_resolve"] = {"solve": rec, "L_diff": rec["L_diff"], "Q_diff": rec["Q_diff"],
                           "dL_rel": rec["L_diff"] / c["final_L"] - 1.0,
                           "dQ_rel": rec["Q_diff"] / c["final_Q"] - 1.0,
                           "L_rel_error_vs_target": rec["L_diff"] / L_TARGET - 1.0}
        print(f"  C W/2 re-solve: L={rec['L_diff'] * 1e12:.1f} pH "
              f"({100 * c['w2_resolve']['dL_rel']:+.2f} %) Q={rec['Q_diff']:.3f} "
              f"({100 * c['w2_resolve']['dQ_rel']:+.2f} %)", flush=True)
        dump()


# ---------------------------------------------------------------------------
# 6. gates

def evaluate_gates(study: dict[str, Any]) -> dict[str, Any]:
    g: dict[str, Any] = {}
    a = study.get("A", {})
    lv = a.get("levels", {})

    # R1: absolute L against the primary referee
    r1: dict[str, Any] = {
        "quantity": "FDFD de-embedded L_diff at 100 MHz vs the PRIMARY referee "
                    "L(strip) - L(short's bridge) on the same square geometry; no FDFD-side "
                    "correction applied",
        "referee_deembedded": study.get("referee", {}).get("deembedded"),
        "referee_strip_only": study.get("referee", {}).get("strip"),
        "per_level_gap": {f"W/{k}": v["L_diff"] / study["referee"]["deembedded"] - 1.0
                          for k, v in lv.items()} if "referee" in study else {},
        "per_level_cells_across_width": {f"W/{k}": v["grid"]["cells_across_width"]
                                         for k, v in lv.items()},
        "d2_small_spiral_gaps": D2_GAPS,
        "criterion": "the W/2 gap must be within %.0f points of D2's small-spiral W/2 gap "
                     "(%.4f), i.e. the deficit at RFIC scale is the known strip "
                     "cross-section discretisation bias and nothing new" % (
                         100 * R1_POINTS, D2_W2_GAP),
        "tolerance_points": R1_POINTS, "d2_w2_gap": D2_W2_GAP}
    aw = a.get("wall", {}).get(str(WALL_PADS[-1]))
    if aw and "1" in lv and "referee" in study:
        r1["wall_correction"] = {
            "what": "the same W/1 model under part A's protocol with the four side walls "
                    "pushed from %d to %d graded cells (%.0f -> %.0f um from the axis; the "
                    "PEC lid is pinned at %d padding cells in both). D2's wall gate on the "
                    "104 um spiral moved L by +0.114 %% for pad 4 -> 8; this device is four "
                    "times larger inside the same margin + padding, so the residual is "
                    "larger and it is measured here rather than assumed." % (
                        WALL_PADS[0], WALL_PADS[-1], lv["1"]["grid"]["wall_x"][1] * 1e6,
                        aw["grid"]["wall_x"][1] * 1e6, PAD_CELLS_Z),
            "pad_cells": [WALL_PADS[0], WALL_PADS[-1]],
            "n_unknowns": [lv["1"]["grid"]["n_unknowns"], aw["grid"]["n_unknowns"]],
            "L_diff": [lv["1"]["L_diff"], aw["L_diff"]],
            "rel_change_L": aw["L_diff"] / lv["1"]["L_diff"] - 1.0,
            "gap_at_pad_4": lv["1"]["gap"], "gap_at_pad_8": aw["gap"],
            "d2_w2_gap": D2_W2_GAP,
            "gap_at_pad_8_minus_d2": aw["gap"] - D2_W2_GAP}
    if "2" in lv and r1["per_level_gap"]:
        r1["w2_gap"] = r1["per_level_gap"]["W/2"]
        r1["w2_gap_minus_d2"] = lv["2"]["gap"] - D2_W2_GAP
        r1["passed"] = abs(r1["w2_gap_minus_d2"]) <= R1_POINTS
    else:
        r1["passed"] = None
    g["R1"] = r1

    # R2: reciprocity, passivity, wall independence
    solves: list[tuple[str, dict[str, Any]]] = []
    for part, key in (("A", "levels"), ("B", "levels"), ("B", "sweep")):
        for k, v in study.get(part, {}).get(key, {}).items():
            if "s_stats" in v:
                solves.append((f"{part}.{key}.{k}", v))
    for k, v in study.get("A", {}).get("wall", {}).items():
        if isinstance(v, dict) and "s_stats" in v:
            solves.append((f"A.wall.{k}", v))
    for k, v in study.get("B", {}).get("wall", {}).items():
        if isinstance(v, dict) and "solve" in v:
            solves.append((f"B.wall.{k}", v["solve"]))
    for k, v in study.get("B", {}).get("t_si_sensitivity", {}).items():
        if isinstance(v, dict) and "solve" in v:
            solves.append((f"B.t_si.{k}", v["solve"]))
    if "w2_resolve" in study.get("C", {}):
        solves.append(("C.w2_resolve", study["C"]["w2_resolve"]["solve"]))
    recip = {n: v["reciprocity_worst_solved"] for n, v in solves}
    passv = {n: v["passivity_worst_solved"] for n, v in solves}
    de_passv = {n: v["s_stats"]["deembedded"]["max_singular_value"] for n, v in solves}
    de_recip = {n: v["s_stats"]["deembedded"]["reciprocity"] for n, v in solves}
    wall = study.get("B", {}).get("wall", {})
    r2: dict[str, Any] = {
        "n_solves_checked": len(solves),
        "reciprocity_worst": max(recip.values()) if recip else None,
        "reciprocity_tolerance": RECIP_TOL,
        "passivity_worst": max(passv.values()) if passv else None,
        "passivity_tolerance": PASSIVITY_TOL,
        "per_solve_reciprocity": recip, "per_solve_passivity": passv,
        "deembedded_note": "reciprocity and passivity are gated on the three SOLVED "
                           "S-matrices (dut, open, short). The de-embedded S is a "
                           "post-processing construct (Y_short - Y_open inverted), not a "
                           "solver output, so its singular values are REPORTED, not gated.",
        "deembedded_passivity_worst": max(de_passv.values()) if de_passv else None,
        "deembedded_reciprocity_worst": max(de_recip.values()) if de_recip else None,
        "wall_pads": list(WALL_PADS), "wall_tolerance": WALL_TOL,
        "wall_rel_change_L": wall.get("rel_change_L"),
        "wall_rel_change_Q": wall.get("rel_change_Q")}
    r2["wall_ladder"] = wall.get("ladder")
    pads_present = [q for q in WALL_PADS if str(q) in wall]
    if len(pads_present) >= 2:
        # the consecutive steps say more than the endpoint gate: a 1/distance
        # extrapolation is only as good as its residual, and if the last step
        # is already tiny the converged value is simply the last point.
        steps = {}
        for lo_p, hi_p in zip(pads_present[:-1], pads_present[1:]):
            steps[f"{lo_p}->{hi_p}"] = (wall[str(hi_p)]["L_diff"]
                                        / wall[str(lo_p)]["L_diff"] - 1.0)
        r2["wall_consecutive_steps_L"] = steps
        r2["wall_last_step_L"] = list(steps.values())[-1]
        r2["wall_interpretation"] = (
            "pad %d -> %d moves L by %+.3f %%, pad %d -> %d by %+.3f %%: the box is "
            "converged from pad %d on, so the wall-free L is the pad-%d value to about "
            "%.2f %% and the pad-%d model used by parts A, B and C is %.2f %% LOW from the "
            "side walls alone. The 1/distance fit over all three points has a %.2f %% "
            "residual and is reported only for completeness." % (
                pads_present[0], pads_present[1], 100 * list(steps.values())[0],
                pads_present[-2], pads_present[-1], 100 * list(steps.values())[-1],
                pads_present[1], pads_present[-1], abs(100 * list(steps.values())[-1]),
                pads_present[0], 100 * (wall[str(pads_present[-1])]["L_diff"]
                                        / wall[str(pads_present[0])]["L_diff"] - 1.0),
                100 * (wall.get("ladder", {}).get("residual_L", float("nan"))))
            if len(pads_present) >= 3 else None)
    r2["passed_reciprocity"] = (r2["reciprocity_worst"] <= RECIP_TOL) if recip else None
    r2["passed_passivity"] = (r2["passivity_worst"] <= PASSIVITY_TOL) if passv else None
    r2["passed_wall"] = (abs(wall["rel_change_L"]) < WALL_TOL
                         if wall.get("rel_change_L") is not None else None)
    subs = [r2["passed_reciprocity"], r2["passed_passivity"], r2["passed_wall"]]
    r2["sub_verdicts"] = {"reciprocity": subs[0], "passivity": subs[1], "wall": subs[2]}
    r2["passed"] = all(subs) if None not in subs else None
    g["R2"] = r2

    # R3: AD vs FD4
    b = study.get("B", {})
    r3: dict[str, Any] = {
        "tolerance": AD_FD_TOL, "fd_order": 4, "fd_step_rel": FD_STEP_REL,
        "checked": {}, "note": "jax.grad of the FDFD metric against a 4th-order central "
                               "difference of the SAME discrete model; 1 %% steps move L by "
                               "~1 %% and Q by ~1 %%, far above the ~1e-9 LU noise floor"}
    pairs = (("dL_dr_out", "r_out", "L", "fd4_dL_diff", 0),
             ("dQ_dwidth", "width", "Q", "fd4_dQ_diff", 2))
    for label, param, which, fdkey, idx in pairs:
        fd = b.get("fd", {}).get(param, {})
        gr = b.get("gradients", {}).get(which, {})
        if fdkey in fd and gr:
            ad = float(gr["grad"][idx])
            fdv = float(fd[fdkey])
            r3["checked"][label] = {"ad": ad, "fd4": fdv,
                                    "rel": abs(ad - fdv) / abs(fdv) if fdv else float("nan")}
    # everything the same stencils also give, reported for free
    extra = {}
    for param, idx in (("r_out", 0), ("width", 2)):
        fd = b.get("fd", {}).get(param, {})
        for which, fdkey in (("L", "fd4_dL_diff"), ("Q", "fd4_dQ_diff")):
            gr = b.get("gradients", {}).get(which, {})
            if fdkey in fd and gr:
                ad, fdv = float(gr["grad"][idx]), float(fd[fdkey])
                extra[f"d{which}_d{param}"] = {
                    "ad": ad, "fd4": fdv,
                    "rel": abs(ad - fdv) / abs(fdv) if fdv else float("nan")}
    r3["all_available"] = extra
    if len(r3["checked"]) == 2:
        r3["worst_rel"] = max(v["rel"] for v in r3["checked"].values())
        r3["passed"] = r3["worst_rel"] <= AD_FD_TOL
    else:
        r3["passed"] = None
    g["R3"] = r3

    # R4: the design target
    c = study.get("C", {})
    r4: dict[str, Any] = {
        "L_target": L_TARGET, "band": L_BAND,
        "paper_context": {"L_diff": 4.000e-9, "Q_diff": Q_PAPER,
                          "what": "arXiv 2607.08852, a 3-turn SYMMETRIC OCTAGONAL spiral; "
                                  "CONTEXT ONLY, never a gate (this DUT is a square, "
                                  "single-ended spiral on a 120 um substrate over a PEC "
                                  "ground)"},
        "complete": bool(c.get("complete"))}
    if c.get("complete"):
        r4.update({"final_theta": c["final_theta"], "final_L": c["final_L"],
                   "final_Q": c["final_Q"], "L_rel_error": c["final_L_rel_error"],
                   "n_objective_evaluations": c.get("n_objective_evaluations"),
                   "n_accepted_iterates": c.get("n_accepted_iterates"),
                   "n_scipy_iterations": c.get("n_scipy_iterations"),
                   "n_replay_mismatches": c.get("n_replay_mismatches"),
                   "feasible": True})
        r4["passed"] = abs(c["final_L_rel_error"]) <= L_BAND
        fdo = c.get("fd_at_optimum", {})
        if "worst_rel" in fdo:
            r4["grad_vs_fd4_worst_rel"] = fdo["worst_rel"]
            r4["grad_vs_fd4_per_param"] = dict(zip(PARAMS, fdo["rel"]))
            r4["grad_vs_fd4_passed"] = fdo["worst_rel"] <= AD_FD_TOL
    else:
        r4["passed"] = None
    prior = study.get("C_attempt_1")
    if prior and prior.get("evals"):
        ev = prior["evals"]
        best = min(ev, key=lambda e: e["loss"])
        r4["prior_attempt"] = {
            "what": "the SAME loop run with lam from the nominal-gradient rule "
                    "(lam = %.4f, e_target = %.0f %%). It is recorded because it is the "
                    "measurement that sized the delivered weight: the loop reduced the loss "
                    "monotonically while moving AWAY from the L target, which is what a "
                    "soft constraint does when its weight is too small -- not a solver "
                    "failure, a weighting error. Stopped after %d evaluations once that was "
                    "clear." % (prior.get("lam", float("nan")), 100 * prior.get("e_target", 0),
                                len(ev)),
            "lam": prior.get("lam"), "n_evals": len(ev),
            "L_rel_error_trajectory": [e["L_rel_error"] for e in ev],
            "Q_trajectory": [e["Q"] for e in ev],
            "loss_trajectory": [e["loss"] for e in ev],
            "best_loss_eval": {"i": best["i"], "theta": best["theta"], "L": best["L"],
                               "Q": best["Q"], "loss": best["loss"],
                               "L_rel_error": best["L_rel_error"]},
            "would_have_passed_R4": abs(best["L_rel_error"]) <= L_BAND,
            "complete": bool(prior.get("complete"))}
    g["R4"] = r4

    # R5: sizes, times, memory
    rows: list[dict[str, Any]] = []
    def add(name: str, rec: dict[str, Any]) -> None:
        grid = rec.get("grid")
        if grid is None:
            return
        rows.append({"name": name, "n_unknowns": grid["n_unknowns"], "shape": grid["shape"],
                     "cells_across_width": grid["cells_across_width"],
                     "seconds_total": rec.get("seconds"),
                     "seconds_per_fixture": rec.get("seconds_per_fixture"),
                     "rss_gb": rec.get("mem", {}).get("rss_gb"),
                     "footprint_gb": rec.get("mem", {}).get("footprint_gb")})
    for part, key in (("A", "levels"), ("B", "levels"), ("B", "sweep")):
        for k, v in study.get(part, {}).get(key, {}).items():
            add(f"{part}.{key}.{k}", v)
    for k, v in a.get("wall", {}).items():
        if isinstance(v, dict) and "grid" in v:
            add(f"A.wall.{k}", v)
    for k, v in study.get("B", {}).get("wall", {}).items():
        if isinstance(v, dict) and "solve" in v:
            add(f"B.wall.{k}", v["solve"])
    for k, v in study.get("B", {}).get("t_si_sensitivity", {}).items():
        if isinstance(v, dict) and "solve" in v:
            add(f"B.t_si.{k}", v["solve"])
    if "w2_resolve" in c:
        add("C.w2_resolve", c["w2_resolve"]["solve"])
    per_fix = [v for r in rows if r["seconds_per_fixture"]
               for v in r["seconds_per_fixture"].values()]
    mems = [r["footprint_gb"] for r in rows if r["footprint_gb"] and
            not math.isnan(r["footprint_gb"])]
    rsss = [r["rss_gb"] for r in rows if r["rss_gb"] and not math.isnan(r["rss_gb"])]
    grad_secs = {k: v["seconds"] for k, v in b.get("gradients", {}).items()}
    loop_secs = [e["seconds"] for e in c.get("evals", [])]
    r5: dict[str, Any] = {
        "what": "largest N, per-fixture factorisation+solve wall time, peak memory per "
                "process; every level actually run. Wall times are UPPER BOUNDS: this "
                "machine ran a second agent throughout.",
        "rows": rows,
        "largest_n_unknowns": max((r["n_unknowns"] for r in rows), default=None),
        "largest_n_unknowns_where": max(rows, key=lambda r: r["n_unknowns"])["name"] if rows
                                    else None,
        "seconds_per_fixture_min": min(per_fix) if per_fix else None,
        "seconds_per_fixture_max": max(per_fix) if per_fix else None,
        "seconds_per_fixture_median": float(np.median(per_fix)) if per_fix else None,
        "peak_footprint_gb": max(mems) if mems else None,
        "peak_rss_gb": max(rsss) if rsss else None,
        "gradient_seconds": grad_secs,
        "loop_eval_seconds_median": float(np.median(loop_secs)) if loop_secs else None,
        "factor_cache_size": int(_MEM["factor_cache_size"]),
        "w3_projection": a.get("w3_projection", {}).get("grid"),
        "solve_units_used": int(_BUDGET["used"]),
        "scaling": b.get("scaling", {}),
        "passed": True if rows else None}

    # the FD stencils go through ONE solve_spiral call for all three fixtures
    # (forward_point) instead of three single-fixture calls (solve_split):
    # same numbers, more peak memory, and it is the path the design loop's
    # value_and_grad takes too, so it is accounted separately.
    single: list[dict[str, Any]] = []
    for param, rec in b.get("fd", {}).items():
        for m, pt in rec.get("points", {}).items():
            if "mem" in pt:
                single.append({"name": f"B.fd.{param}:{m}", "seconds": pt["seconds"],
                               "footprint_gb": pt["mem"]["footprint_gb"],
                               "rss_gb": pt["mem"]["rss_gb"]})
    for m, pt in c.get("fd_at_optimum", {}).get("points", {}).items():
        if isinstance(pt, dict) and "mem" in pt:
            single.append({"name": f"C.fd_at_optimum.{m}", "seconds": pt["seconds"],
                           "footprint_gb": pt["mem"]["footprint_gb"],
                           "rss_gb": pt["mem"]["rss_gb"]})
    if single:
        fps = [r["footprint_gb"] for r in single if r["footprint_gb"]
               and not math.isnan(r["footprint_gb"])]
        r5["single_call_path"] = {
            "what": "one spiral.solve_spiral call for all three fixtures (forward_point: the "
                    "FD stencils, and the same path the design loop's value_and_grad takes) "
                    "against solve_split's three single-fixture calls. The metrics are "
                    "BITWISE the same (gated in tests/unit/fdfd/test_fdfd_rfic_spiral.py); the peak "
                    "memory is not, because the three fixtures' assembled operators are "
                    "live in one XLA program.",
            "n_unknowns": (lv.get("1") or {}).get("grid", {}).get("n_unknowns"),
            "n_points": len(single), "rows": single,
            "peak_footprint_gb": max(fps) if fps else None,
            "seconds_min": min(r["seconds"] for r in single),
            "seconds_max": max(r["seconds"] for r in single)}
        if fps and r5["peak_footprint_gb"] is not None:
            r5["peak_footprint_gb_any_path"] = max(r5["peak_footprint_gb"], max(fps))

    # what the 3-cells-across-W level would cost HERE: the N is exact (the
    # model is built), the cost is the measured cost law of this very fixture
    # family extrapolated, bracketed by the textbook 3-D direct-solver
    # exponents (work ~ N^2, fill ~ N^(4/3)).
    sc = b.get("scaling", {})
    w3 = a.get("w3_projection", {}).get("grid")
    anchor = sc.get("points", {}).get(str(PAD_CELLS))
    if w3 and anchor and "exponent_time" in sc:
        ratio = w3["n_unknowns"] / anchor["n_unknowns"]
        per_fix = anchor["seconds"]
        fac_gb = anchor["footprint_after_gb"]
        r5["w3_cost_projection"] = {
            "what": "extrapolation, NOT a measurement: the W/3 model is built (N exact) but "
                    "never solved on this machine",
            "n_unknowns": w3["n_unknowns"], "n_unknowns_ratio": ratio,
            "anchor": {"n_unknowns": anchor["n_unknowns"], "seconds_per_fixture": per_fix,
                       "peak_footprint_gb": fac_gb,
                       "memory_basis": sc.get("memory_basis")},
            "measured_law": {"exponent_time": sc["exponent_time"],
                             "exponent_memory": sc["exponent_memory"],
                             "seconds_per_fixture": per_fix * ratio ** sc["exponent_time"],
                             "seconds_three_fixtures": 3.0 * per_fix * ratio ** sc["exponent_time"],
                             "peak_footprint_gb": fac_gb * ratio ** sc["exponent_memory"]},
            "textbook_law": {"exponent_time": 2.0, "exponent_memory": 4.0 / 3.0,
                             "seconds_per_fixture": per_fix * ratio ** 2.0,
                             "seconds_three_fixtures": 3.0 * per_fix * ratio ** 2.0,
                             "peak_footprint_gb": fac_gb * ratio ** (4.0 / 3.0)},
            "machine_ram_gb": 38.0}
    g["R5"] = r5
    return g


# ---------------------------------------------------------------------------
# 7. figure

# Okabe & Ito's colourblind-safe qualitative palette (Okabe and Ito 2008),
# used in a FIXED order and never cycled; the study's three recurring roles
# are FDFD = blue, referee / reference lines = vermillion, secondary
# reference = bluish green. No panel carries two y-scales: where two
# quantities of different units belong together they get two panels, or are
# indexed to a common base (panel h).
C_FDFD = "#0072B2"
C_REF = "#D55E00"
C_SECOND = "#009E73"
C_THIRD = "#CC79A7"
C_INK = "#2b2b2b"
C_MUTED = "#6e6e6e"


def _ax_style(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=9, color=C_INK)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.tick_params(labelsize=7.5, colors=C_MUTED, length=3)
    ax.grid(alpha=0.25, lw=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c8c8c8")


def figure(study: dict[str, Any], path: pathlib.Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 4, figsize=(19.0, 8.6))
    ref = study.get("referee", {})
    a_lv = study.get("A", {}).get("levels", {})
    b = study.get("B", {})
    c = study.get("C", {})

    # (a) geometry and the grid that is the whole resolution story
    axx = ax[0, 0]
    try:
        model = build_level(1.0)
        for poly in [q for ps in model.spiral.polygons.values() for q in ps]:
            q = np.asarray(poly) * 1e6
            axx.fill(q[:, 0], q[:, 1], color=C_FDFD, alpha=0.35, lw=0.5, ec=C_FDFD)
        for v in np.asarray(model.x_nom) * 1e6:
            axx.axvline(v, lw=0.25, color="#d5d5d5", zorder=0)
        for v in np.asarray(model.y_nom) * 1e6:
            axx.axhline(v, lw=0.25, color="#d5d5d5", zorder=0)
        for col in model.columns:
            axx.plot([model.x_nom[col.i0] * 1e6, model.x_nom[col.i1] * 1e6],
                     [model.y_nom[col.j0] * 1e6] * 2, color=C_REF, lw=2.5,
                     solid_capstyle="butt")
        g = model_record(model)
        pad = model.spec.pad_cells
        inner = np.diff(np.asarray(model.x_nom)[pad:len(model.x_nom) - pad]) * 1e6
        axx.set_title(f"(a) square 3-turn, r_out {R_OUT * 1e6:.0f}, W {WIDTH * 1e6:.0f}, "
                      f"S {SPACING * 1e6:.0f} um\nbase_dx = W: N = {g['n_unknowns']}, "
                      f"meshed cells {inner.min():.0f}-{inner.max():.0f} um, "
                      f"{g['cells_across_width']} across W", fontsize=9, color=C_INK)
        axx.text(0.98, 0.02, "red: the two lead-column\nport footprints", fontsize=6.5,
                 color=C_MUTED, ha="right", va="bottom", transform=axx.transAxes)
    except Exception as exc:                     # pragma: no cover - plotting only
        axx.text(0.5, 0.5, f"geometry: {exc}", ha="center", fontsize=7)
    axx.set_xlabel("x (um)", fontsize=8)
    axx.set_ylabel("y (um)", fontsize=8)
    axx.tick_params(labelsize=7.5, colors=C_MUTED, length=3)
    axx.set_xlim(-265, 265)
    axx.set_ylim(-305, 265)
    axx.set_aspect("equal")

    # (b) part A: absolute L against the referee -- zoomed, because the
    # question is a 13 % gap, not whether L is positive
    axx = ax[0, 1]
    if a_lv and ref:
        keys = sorted(a_lv, key=float)
        xs = np.arange(len(keys) + (1 if study.get("A", {}).get("wall") else 0))
        vals = [a_lv[k]["L_diff"] * 1e9 for k in keys]
        labels = [f"base_dx=W/{k}\npad 4, N={a_lv[k]['grid']['n_unknowns']}" for k in keys]
        aw = study.get("A", {}).get("wall", {}).get(str(WALL_PADS[-1]))
        if aw:
            vals.append(aw["L_diff"] * 1e9)
            labels.append(f"base_dx=W\npad {WALL_PADS[-1]}, N={aw['grid']['n_unknowns']}")
        axx.bar(xs, vals, 0.55, color=C_FDFD, label="FDFD de-embedded L_diff")
        axx.axhline(ref["deembedded"] * 1e9, color=C_REF, lw=2.0,
                    label="referee L(strip) - L(bridge)")
        axx.axhline(ref["mohan_wheeler"]["L"] * 1e9, color=C_THIRD, lw=1.8, ls=":",
                    label="Mohan-Wheeler closed form (no ground)")
        axx.axhline(ref["strip_no_ground"] * 1e9, color=C_SECOND, lw=1.2, ls="--",
                    label="referee strip, no ground")
        axx.axhline(4.000, color=C_MUTED, lw=1.2, ls="-.",
                    label="paper 4.000 nH (context only)")
        for x, v in zip(xs, vals):
            axx.text(x, v + 0.02, f"{100 * (v * 1e-9 / ref['deembedded'] - 1):+.2f} %",
                     ha="center", fontsize=7.5, color=C_INK)
        axx.set_xticks(xs)
        axx.set_xticklabels(labels, fontsize=7)
        axx.set_ylim(2.8, 4.35)
        axx.legend(fontsize=6.2, loc="lower left", framealpha=0.92)
    _ax_style(axx, "(b) A: absolute L at 100 MHz, uniform-current metal", "",
              "L_diff (nH)")

    # (c) and (d) the frequency sweep -- two panels, not two y-scales
    sw = b.get("sweep", {})
    if sw:
        keys = sorted(sw, key=float)
        f = np.array([float(k) for k in keys])
        L = np.array([sw[k]["L_diff"] for k in keys]) * 1e9
        Q = np.array([sw[k]["Q_diff"] for k in keys])
        axx = ax[0, 2]
        axx.plot(f, L, "o-", color=C_FDFD, lw=2.0, ms=8, label="L_diff")
        axx.axvline(2.45, color=C_MUTED, lw=1.0, ls=":")
        axx.text(2.45, 0.02, " 2.45 GHz", fontsize=6.5, color=C_MUTED,
                 transform=axx.get_xaxis_transform(), va="bottom")
        for xv, yv in zip(f, L):
            axx.annotate(f"{yv:.2f}", (xv, yv), textcoords="offset points",
                         xytext=(0, 9), ha="center", fontsize=6.5, color=C_INK)
        _ax_style(axx, "(c) B: L_diff vs frequency (base_dx = W)", "frequency (GHz)",
                  "L_diff (nH)")
        axx.legend(fontsize=7, loc="upper left")
        axx = ax[0, 3]
        axx.plot(f, Q, "s-", color=C_REF, lw=2.0, ms=8, label="Q_diff")
        axx.axvline(2.45, color=C_MUTED, lw=1.0, ls=":")
        axx.axhline(Q_PAPER, color=C_MUTED, lw=1.2, ls="-.",
                    label=f"paper {Q_PAPER:.2f} (context only)")
        for xv, yv in zip(f, Q):
            axx.annotate(f"{yv:.2f}", (xv, yv), textcoords="offset points",
                         xytext=(0, 9), ha="center", fontsize=6.5, color=C_INK)
        _ax_style(axx, "(d) B: Q_diff vs frequency (Leontovich sheet, delta/t = 0.61-0.92)",
                  "frequency (GHz)", "Q_diff")
        axx.legend(fontsize=6.5, loc="lower center")

    # (e) the substrate-thickness deviation
    axx = ax[1, 0]
    ts = b.get("t_si_sensitivity", {})
    if ref.get("ground_image_curve"):
        cur = ref["ground_image_curve"]
        keys = sorted(cur, key=float)
        axx.plot([cur[k]["t_si"] * 1e6 for k in keys],
                 [cur[k]["deembedded"] * 1e9 for k in keys], "o--", color=C_REF, lw=1.8,
                 ms=8, label="referee (ground image), 100 MHz")
    if ts.get("chosen"):
        xs = [ts["alt"]["t_si"] * 1e6, ts["chosen"]["t_si"] * 1e6]
        axx.plot(xs, [ts["alt"]["L_diff"] * 1e9, ts["chosen"]["L_diff"] * 1e9], "s-",
                 color=C_FDFD, lw=2.0, ms=8, label="FDFD L_diff, 2.45 GHz")
        axx.annotate(f"{100 * ts['dL_rel']:+.1f} % (referee "
                     f"{100 * ts['referee_dL_rel_same_pair']:+.1f} %)",
                     (xs[0], ts["alt"]["L_diff"] * 1e9), textcoords="offset points",
                     xytext=(6, -12), fontsize=6.8, color=C_INK)
    axx.axvline(T_SI_REAL * 1e6, color=C_MUTED, ls=":", lw=1.2)
    axx.text(T_SI_REAL * 1e6, 0.06, " real die ~190 um", fontsize=6.8, color=C_MUTED,
             transform=axx.get_xaxis_transform(), va="bottom")
    axx.axvline(T_SI * 1e6, color=C_FDFD, ls=":", lw=1.0)
    axx.text(T_SI * 1e6, 0.06, "chosen 120 um ", fontsize=6.8, color=C_FDFD, ha="right",
             transform=axx.get_xaxis_transform(), va="bottom")
    _ax_style(axx, "(e) the substrate-thickness deviation, quantified",
              "silicon over the PEC ground (um)", "L (nH)")
    axx.legend(fontsize=6.8, loc="lower right")

    # (f) and (g) the design loop -- again two panels, one scale each
    evals = c.get("evals", [])
    if evals:
        i = [e["i"] for e in evals]
        axx = ax[1, 1]
        axx.axhspan(L_TARGET * (1 - L_BAND) * 1e9, L_TARGET * (1 + L_BAND) * 1e9,
                    color=C_REF, alpha=0.12, label=f"target +- {100 * L_BAND:.0f} %")
        axx.axhline(L_TARGET * 1e9, color=C_REF, lw=1.8,
                    label=f"target {L_TARGET * 1e9:.3f} nH")
        axx.plot(i, [e["L"] * 1e9 for e in evals], "o-", color=C_FDFD, lw=1.6, ms=5.5,
                 label="L_diff per evaluation")
        acc = [e for e in evals if e.get("accepted")]
        if acc:
            axx.plot([e["i"] for e in acc], [e["L"] * 1e9 for e in acc], "o",
                     color=C_FDFD, ms=10, mfc="none", mew=1.8,
                     label="accepted L-BFGS-B iterate")
        _ax_style(axx, "(f) C: the design loop, L (one solve + adjoint each)",
                  "objective evaluation", "L_diff (nH)")
        axx.legend(fontsize=6.5, loc="best")
        axx = ax[1, 2]
        axx.plot(i, [e["Q"] for e in evals], "s-", color=C_REF, lw=1.6, ms=5.5,
                 label="Q_diff per evaluation")
        if c.get("Q_at_nominal"):
            axx.axhline(c["Q_at_nominal"], color=C_MUTED, lw=1.2, ls="--",
                        label=f"Q at the nominal ({c['Q_at_nominal']:.3f})")
        _ax_style(axx, "(g) C: the design loop, Q (maximised subject to L)",
                  "objective evaluation", "Q_diff")
        axx.legend(fontsize=6.5, loc="best")

    # (h) cost: both measures INDEXED to the base_dx = W solve, so one scale
    axx = ax[1, 3]
    sc = b.get("scaling", {})
    if sc.get("n_unknowns"):
        n = np.asarray(sc["n_unknowns"], dtype=float)
        s = np.asarray(sc["seconds"], dtype=float)
        m = np.asarray(sc["footprint_after_gb"], dtype=float)
        n0 = n[-1]
        axx.plot(n / n0, s / s[-1], "o", color=C_FDFD, ms=8,
                 label=f"wall seconds / fixture (N^{sc['exponent_time']:.2f})")
        axx.plot(n / n0, m / m[-1], "^", color=C_REF, ms=8,
                 label=f"peak footprint (N^{sc['exponent_memory']:.2f})")
        xs = np.logspace(np.log10((n / n0).min()), np.log10(7.0), 50)
        axx.plot(xs, xs ** sc["exponent_time"], "-", color=C_FDFD, lw=1.2, alpha=0.7)
        axx.plot(xs, xs ** sc["exponent_memory"], "-", color=C_REF, lw=1.2, alpha=0.7)
        w3 = study.get("gates", {}).get("R5", {}).get("w3_cost_projection")
        if w3:
            r = w3["n_unknowns_ratio"]
            axx.axvline(r, color=C_MUTED, ls=":", lw=1.2)
            axx.text(r * 0.95, 1.5, f"base_dx = W/3\nN = {w3['n_unknowns']}\n"
                                    f"{w3['measured_law']['seconds_three_fixtures'] / 3600:.1f} h"
                                    f" / solve, {w3['measured_law']['peak_footprint_gb']:.0f} GB"
                                    f"\n(not run: {w3['machine_ram_gb']:.0f} GB machine)",
                     fontsize=6.5, color=C_MUTED, ha="right", va="bottom")
        axx.set_xscale("log")
        axx.set_yscale("log")
        axx.axhline(1.0, color="#d5d5d5", lw=0.8)
        _ax_style(axx, "(h) R5: measured cost law, indexed to the base_dx = W solve\n"
                       f"(N = {int(n0)}: {s[-1]:.0f} s/fixture, {m[-1]:.1f} GB)",
                  "N / N(base_dx = W)", "cost / cost at base_dx = W")
        axx.legend(fontsize=6.5, loc="upper left")

    fig.suptitle("Study H: an RFIC-scale square spiral (published geometry, SG13G2-like "
                 "stack) through the differentiable FDFD pipeline", fontsize=11,
                 color=C_INK)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 8. main

def fixture_record() -> dict[str, Any]:
    return {
        "dut": "SQUARE 3-turn spiral (rfx.fdfd.gds.rect_spiral) on TopMetal2 with a "
               "TopMetal1 underpass -- NOT the paper's octagon, see the module doc",
        "n_turns": N_TURNS, "r_out": R_OUT, "spacing": SPACING, "width": WIDTH,
        "lead": LEAD, "margin": MARGIN, "theta0": list(THETA0), "params": list(PARAMS),
        "a_in": R_OUT - 0.5 * WIDTH - N_TURNS * (WIDTH + SPACING),
        "feasibility_margin": feasibility_margin(THETA0),
        "stack": {"t_si": T_SI, "t_si_alt": T_SI_ALT, "t_si_real_die": T_SI_REAL,
                  "t_ox_low": T_OX_LOW, "t_m1": T_M1, "t_via": T_VIA, "t_m2": T_M2,
                  "t_ox_high": T_OX_HIGH, "t_air": T_AIR, "base_dz": BASE_DZ,
                  "metal_cells": METAL_CELLS, "eps_si": EPS_SI, "eps_ox": EPS_OX,
                  "ground_height_to_m2_centre": GROUND_H, "dz_underpass": DZ_UNDER},
        "grid": {"pad_cells": PAD_CELLS, "pad_cells_z": PAD_CELLS_Z, "pad_ratio": PAD_RATIO,
                 "port_gap_cells": 1, "divs_run": list(DIVS), "div_reported_only": DIV_W3},
        "part_a": {"freq": FREQ_A, "sigma_volumetric": SIGMA_VOL, "dielectrics": "vacuum"},
        "part_b": {"freq": FREQ_B, "sigma_metal": SIGMA_CU, "sigma_si": SIGMA_SI,
                   "sweep_freqs": list(SWEEP_FREQS)},
        "fd_step_rel": FD_STEP_REL,
    }


def meta_record() -> dict[str, Any]:
    return {
        "paper": {"ref": "arXiv 2607.08852, 2.4 GHz LC-VCO, IHP SG13G2",
                  "geometry": "3-turn SYMMETRIC OCTAGONAL spiral, r_out 218 um, W 30 um, "
                              "S 14 um, TopMetal2 with a TopMetal1 underpass",
                  "reported": {"L_diff": 4.000e-9, "Q_diff": Q_PAPER, "freq": 2.45e9},
                  "status": "CONTEXT ONLY -- no gate in this study compares to it"},
        "deviations": [
            {"what": "square spiral instead of the paper's octagon",
             "why": "the differentiable body-fitted metric of rfx.fdfd.spiral is built on "
                    "axis-aligned breakpoints (rect_spiral); gds.octagonal_spiral exists but "
                    "spiral.py has no breakpoint map for it",
             "effect": "a square of the same r_out has more metal and a longer conductor "
                       "than the inscribed octagon, so its L is HIGHER at equal r_out; it is "
                       "a different device, not a discretisation of the paper's"},
            {"what": "single-ended spiral with an underpass instead of a symmetric "
                     "(two-winding) differential inductor",
             "why": "rect_spiral builds one winding; SpiralSpec has no symmetric topology",
             "effect": "L_diff here is the series inductance of one winding between two "
                       "terminals, not the differential inductance of two interleaved ones"},
            {"what": "silicon 120 um thick over a PEC ground instead of ~190 um",
             "why": "the vertical grid is layer-uniform (gds.z_lines) and every 60 um "
                    "silicon cell costs 3 x 43 x 44 unknowns",
             "effect": "MEASURED, both by the FDFD (two thicknesses at the coarse level) and "
                       "by the referee's ground image at 60/120/190/300 um: see "
                       "B.t_si_sensitivity"},
            {"what": "TopVia2 widened from ~0.85 um to 2 um",
             "why": "leontovich_validity divides the skin depth by the THINNEST metal slab; "
                    "at 0.85 um the via, a 30 um square carrying the series current between "
                    "the layers, would set that number instead of the 2 um underpass",
             "effect": "the M2-M1 centre separation becomes 4.5 um instead of 3.35 um "
                       "(+34 %), slightly changing the underpass mutual"},
            {"what": "passivation permittivity lumped into the oxide's 4.1 (public 6.6)",
             "why": "SmallStack has one oxide permittivity for the whole back end",
             "effect": "a 1.5 um layer directly above the strip is 4.1 instead of 6.6: the "
                       "strip's shunt capacitance to the air side is slightly low, which "
                       "raises the self-resonance a little"},
            {"what": "the Leontovich sheet is used outside its delta << t range",
             "why": "the pipeline has no finite-thickness skin-effect metal model",
             "effect": "absolute Q at 2.45 GHz is not a validated number: see "
                       "B.leontovich for delta/t"},
        ],
        "calibration": {
            "what": "the measurement that forced linear_solve.factor_cache_size(1)",
            "n_unknowns": 85785, "shape": [43, 44, 14],
            "three_fixture_solve_seconds": 216.7266721725464,
            "peak_footprint_gb_with_default_cache_of_4": 15.287808179855347,
            "L_diff_pH_at_100MHz_t_si_60um": 2624.9959446102694,
            "note": "t_si = 60 um configuration, run once before the study to size it; with "
                    "three LU factors live the footprint is 15.3 GB, which is why every "
                    "solve in the study keeps ONE",
        },
        "protocol_source": "validation/fdfd/spiral_convergence.py (study D2): "
                           "uniform-current volumetric metal sigma = 3e6 at 100 MHz, vacuum "
                           "dielectrics, pad_cells = 4 graded wall cells, area-exact corner "
                           "convention, primary referee L(strip) - L(short's bridge)",
        "python": sys.version.split()[0],
    }


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__ and __doc__.splitlines()[0])
    ap.add_argument("--json", type=pathlib.Path, default=JSON_PATH)
    ap.add_argument("--png", type=pathlib.Path, default=PNG_PATH)
    ap.add_argument("--from-json", action="store_true",
                    help="reassemble from the JSON on disk and only do what is missing")
    ap.add_argument("--max-solves", type=int, default=None,
                    help="stop cleanly after this many solve units and exit %d"
                         % EXIT_INCOMPLETE)
    ap.add_argument("--blocks", default="referee,a,b,c",
                    help="comma-separated subset of referee,a,b,c")
    ap.add_argument("--only", default="",
                    help="comma-separated subset of part B's items (%s); empty = all. Lets "
                         "two processes work on disjoint items into two JSON files."
                         % ",".join(B_ITEMS))
    ap.add_argument("--a-items", default="",
                    help="comma-separated subset of part A's items (%s); empty = all"
                         % ",".join(A_ITEMS))
    ap.add_argument("--c-items", default="",
                    help="comma-separated subset of part C's items (%s); empty = all"
                         % ",".join(C_ITEMS))
    ap.add_argument("--fd-params", default="",
                    help="comma-separated subset of %s: which optimum-FD stencils this "
                         "process does; empty = all" % ",".join(PARAMS))
    ap.add_argument("--merge", type=pathlib.Path, action="append", default=[],
                    help="fold a sibling run's JSON into this one (keys this JSON already "
                         "has always win); repeatable")
    ap.add_argument("--factor-cache", type=int, default=1,
                    help="rfx.fdfd.linear_solve.factor_cache_size: how many LU factors are "
                         "kept. 1 (the default here) bounds a solve process at ONE factor, "
                         "which is what made these sizes fit; 3 lets a reverse pass reuse "
                         "the forward factorisations (measured ~1.9x faster per "
                         "value_and_grad, ~1.7x the peak footprint) and is worth using when "
                         "nothing else is running on the machine. It changes cost only -- "
                         "the solves are deterministic and the numbers are identical.")
    ap.add_argument("--c-key", default="C",
                    help="top-level JSON key part C writes to (default C); used to park an "
                         "earlier attempt beside the delivered one")
    ap.add_argument("--lam", type=float, default=None,
                    help="explicit penalty weight for part C's objective, overriding the "
                         "nominal-gradient rule; the override and its justification are "
                         "recorded in the JSON")
    ap.add_argument("--rename", action="append", default=[],
                    help="OLD:NEW -- rename a top-level JSON key before working (scripted, "
                         "so parking an earlier attempt is reproducible); repeatable")
    ap.add_argument("--redo", action="append", default=[],
                    help="dotted JSON path to DROP before working (e.g. B.scaling), so that "
                         "block is measured again; repeatable")
    ap.add_argument("--no-figure", action="store_true")
    args = ap.parse_args(argv)
    for name in (n.strip() for n in args.only.split(",") if n.strip()):
        if name not in B_ITEMS:
            ap.error(f"--only: unknown item {name!r} (one of {B_ITEMS})")
        _ONLY.add(name)
    for name in (n.strip() for n in args.a_items.split(",") if n.strip()):
        if name not in A_ITEMS:
            ap.error(f"--a-items: unknown item {name!r} (one of {A_ITEMS})")
        _ONLY_A.add(name)
    for name in (n.strip() for n in args.c_items.split(",") if n.strip()):
        if name not in C_ITEMS:
            ap.error(f"--c-items: unknown item {name!r} (one of {C_ITEMS})")
        _ONLY_C.add(name)
    for name in (n.strip() for n in args.fd_params.split(",") if n.strip()):
        if name not in PARAMS:
            ap.error(f"--fd-params: unknown parameter {name!r} (one of {PARAMS})")
        _FD_PARAMS.add(name)

    import jax
    jax.config.update("jax_enable_x64", True)
    from rfx.fdfd.linear_solve import factor_cache_size
    factor_cache_size(args.factor_cache)          # see the module doc and --factor-cache
    _MEM["factor_cache_size"] = int(args.factor_cache)

    study: dict[str, Any] = {}
    if args.from_json and args.json.exists():
        study = json.loads(args.json.read_text())
    study["meta"] = meta_record()
    study["fixture"] = fixture_record()
    study.setdefault("seconds", 0.0)
    t_start = time.time()

    def dump() -> None:
        study["seconds"] = study.get("seconds_prior", 0.0) + time.time() - t_start
        study["gates"] = evaluate_gates(study)
        study["solve_units_used_total"] = study.get("solve_units_prior", 0) + _BUDGET["used"]
        tmp = args.json.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(study, indent=1, allow_nan=True))
        tmp.replace(args.json)

    global _C_KEY, _LAM
    _C_KEY = args.c_key
    _LAM = args.lam
    for spec in args.rename:
        old_k, _, new_k = spec.partition(":")
        if old_k in study:
            study[new_k] = study.pop(old_k)
            print(f"  --rename: {old_k} -> {new_k}", flush=True)
        else:
            print(f"  --rename: {old_k} not present", flush=True)
    for dotted in args.redo:
        node: Any = study
        parts = dotted.split(".")
        for key in parts[:-1]:
            node = node.get(key) if isinstance(node, dict) else None
            if node is None:
                break
        if isinstance(node, dict) and parts[-1] in node:
            node.pop(parts[-1])
            print(f"  --redo: dropped {dotted}", flush=True)
        else:
            print(f"  --redo: {dotted} not present, nothing dropped", flush=True)
    for path in args.merge:
        if not path.exists():
            raise SystemExit(f"--merge {path}: no such file")
        src = json.loads(path.read_text())
        n = merge_missing(study, src)
        study.setdefault("merged_runs", []).append(
            {"path": path.name, "keys_added": n, "seconds": src.get("seconds"),
             "solve_units": src.get("solve_units_used_total"),
             "what": "a sibling process that worked on disjoint items of this study "
                     "(--only / --c-items / --fd-params) in parallel; its wall time is NOT "
                     "added to this study's `seconds`, which is per-process"})
        print(f"  merged {n} keys from {path}", flush=True)
    study["seconds_prior"] = float(study.get("seconds", 0.0))
    study["solve_units_prior"] = int(study.get("solve_units_used_total", 0))
    _BUDGET["max"] = args.max_solves
    blocks = [b.strip() for b in args.blocks.split(",") if b.strip()]
    incomplete = False
    try:
        if "referee" in blocks:
            study["referee"] = referee_block()      # host numpy, ~2 s: never cached
            r = study["referee"]
            print(f"  referee: strip {r['strip'] * 1e9:.4f} nH  bridge "
                  f"{r['bridge'] * 1e12:.2f} pH  PRIMARY {r['deembedded'] * 1e9:.4f} nH  "
                  f"(Mohan-Wheeler {r['mohan_wheeler']['L'] * 1e9:.4f} nH)", flush=True)
            dump()
        if "a" in blocks:
            part_a(study, dump)
        if "b" in blocks:
            part_b(study, dump)
        if "c" in blocks:
            part_c(study, dump)
    except BudgetStop:
        incomplete = True
    dump()
    if not args.no_figure:
        try:
            figure(study, args.png)
        except Exception as exc:                  # pragma: no cover - plotting only
            print(f"  figure failed: {exc}", flush=True)
    g = study["gates"]
    print("\ngates:", flush=True)
    for k in sorted(g):
        p = g[k].get("passed")
        print(f"  {k}: {'PASS' if p else ('FAIL' if p is False else 'incomplete')}", flush=True)
    print(f"wrote {args.json} ({args.json.stat().st_size} bytes), "
          f"{_BUDGET['used']} solve units this chunk", flush=True)
    if incomplete:
        return EXIT_INCOMPLETE
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
