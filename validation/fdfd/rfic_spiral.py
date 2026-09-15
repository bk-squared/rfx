"""Study R, parts A and B: the paper-scale RFIC inductor through the whole
differentiable FDFD pipeline on the GPU (cuDSS), at a resolved grid.

THE DEVICE, AND WHAT THE PAPER NUMBERS ARE HERE
------------------------------------------------
The 2.4 GHz LC-VCO paper (arXiv 2607.08852, IHP SG13G2) reports L_diff =
4.000 nH and Q_diff = 16.80 at 2.45 GHz for a 3-turn SYMMETRIC OCTAGONAL
spiral with r_out = 218 um, W = 30 um, S = 14 um. This study solves a SQUARE,
single-ended 3-turn spiral with the same (r_out, W, S) (``rfx.fdfd.spiral`` is
built on ``gds.rect_spiral``): a square of the same r_out has more conductor,
so its L is higher, and the paper's numbers are CONTEXT ONLY -- no gate
compares to them. Part C (``rfic_design.py``) uses 4.000 nH as a design
target.

THE REVIEW OF THE PREVIOUS STUDY H (``prior_H_review`` in the JSON)
------------------------------------------------------------------
The previous rfic_spiral.py / .json (git a9c35192) were built by an agent that
died before reporting. What stands: its reciprocity / passivity numbers
(1.97e-10, 1 + 1.07e-10 over 12 solves) and its AD-vs-FD4 checks (1.4e-8,
5.7e-8) -- for its own coarse model only. What does not: R1 (100 MHz,
3e6 S/m, a 29 um skin depth on a 30 um strip, a cell-defined fixture whose
"W/1" and "W/2" were one grid with 2 cells across W; superseded by the
ladder's paper block), the 2.45 GHz values (2 cells across W, 120 um uniform
silicon, walls 1 % close), and R4: its docstring's design numbers exist
nowhere on disk (the JSON has no part-C record) and are discarded. WHY R2
FAILED: not reciprocity or passivity, which passed, but a third sub-gate,
side-wall independence -- pushing the cell-defined walls from 4 to 8 graded
cells moved L_diff by +1.0245 % against < 1 % (steps +0.983 % then +0.041 %).
Here the walls are physical, 10 W from the DUT (the ladder's rule).

THE FIXTURE (``fixture`` in the JSON)
-------------------------------------
The level-invariant fixture of ``rfx.fdfd.spiral`` (the ladder track's
options): walls and lid 10 W = 300 um from the DUT box, the short standard's
post 30 um from each lead column, a physical 60 um port gap, nested joint
refinement of a level-1 mesh (level m: every level-1 interval in x, y and z
cut into m). Level 1 has 2 cells across W, level 2 (the RESOLVED level) 4,
level 3 6. Stack (study H's SG13G2-like numbers): silicon eps 11.9, 2 S/m;
oxide eps 4.1; TopMetal1 2 um (underpass), TopVia2 2 um, TopMetal2 3 um;
every metal cell a Leontovich sheet of 3.05e7 S/m; PEC ground under the
silicon. 2.45 GHz.

THE GRADED SUBSTRATE. ``rfx.fdfd.spiral`` makes the silicon layer-uniform;
the study grades it without editing that file (``graded_substrate``: the
invariant fixture's mandatory-z-line hook is fed a ``pad_to`` grading, pinned
to the port-gap line, for the duration of one build). Chosen thickness
190 um. Grading ratio 2, not 1.5, because of the 24 GB card: cuDSS's plan of
the resolved level is 22.38 GB of factor at 1.5 (it died with ALLOC_FAILED)
and 20.51 GB at 2 (N = 772892). At level 1 the graded silicon (6 cells)
raises L_diff by 1.02 % over the layer-uniform one (4 cells) and ratio 1.5
(7 cells) moves it a further +0.15 %; Q moves < 0.06 %.

RESULTS (every number is in ``rfic_spiral.json``; GPU runs in ``runs``)
----------------------------------------------------------------------
A. 2.45 GHz. Level 1 (N = 100126): L_diff = 3294.16 pH, Q_diff = 21.059,
   L_se = 3440.89 pH, Q_se = 13.623. Level 2 (N = 772892, 77 s per
   three-fixture forward on the RTX 4090, 22.54 GB in use): L_diff =
   3506.98 pH, Q_diff = 23.631, L_se = 3661.39 pH, Q_se = 14.763. From 2 to
   4 cells across W, L_diff rises 6.46 % and Q_diff 12.2 %: the grid is not
   converged in absolute terms (see ACCURACY). The level-3 grid (6 cells,
   N = 2577198) is a 106.71 GB factor (plan) and fits no card this study may
   use; part C (``rfic_design.py`` R7) measures the one-level-finer shift in
   its in-plane and vertical halves instead.
R2 every S-matrix of the 35 solves of this file (140: the three solved
   fixtures and the de-embedded one each; 27 GPU solves through cuDSS and
   the 8 CPU solves of WALLS and REMESH through SuperLU): |S12 - S21| <=
   6.45e-11 (the GPU solves alone 2.06e-12), largest singular value <=
   0.99904 (strictly passive: lossy silicon and sheet). PASSES (<= 1e-8,
   <= 1 + 1e-8).
Sweep at level 2, 0.5 / 1 / 2 / 2.45 / 3.5 / 5 / 6.5 / 8 GHz: L_diff =
   3495.91 / 3448.51 / 3470.87 / 3506.98 / 3645.92 / 4001.51 / 4644.50 /
   5872.72 pH, Q_diff = 17.020 / 21.078 / 23.674 / 23.631 / 22.158 / 18.462 /
   14.122 / 9.731. L dips to its minimum at 1 GHz (the sheet's internal
   inductance falls like f^-1/2) and then rises with the self-resonance: a
   lumped fit 1/L = (1 - (f/f0)^2)/L0 over f >= 2.45 GHz gives f0 = 12.25 GHz
   (L0 = 3351.28 pH, worst residual 0.56 %) -- a trend, not a resolved
   resonance. Q peaks between 2 and 2.45 GHz.
Substrate thickness (the ground image is the independent reference: the
   silicon is non-magnetic and its 2.45 GHz skin depth at 2 S/m is 7.2 mm):
   at level 1, 120 / 300 um move L_diff by -6.06 % / +2.64 % from 190 um
   (Greenhouse referee's ground image: -5.86 % / +2.91 %) and Q_diff by
   -3.46 % / +6.70 %; at level 2, 120 um moves L_diff -5.79 % (referee
   -5.86 %) and Q_diff -4.21 %. The thickness is a first-order parameter:
   190 um is study H's "real die", not a measured number of the paper.
B. Gradients at level 2 (one forward + two ``jax.vjp`` reverse passes through
   cuDSS, 228 s): dL_diff/d(r_out, spacing, width) = (3.5989e-05,
   -6.7839e-05, -1.1531e-04) H/m, dQ_diff/d(...) = (-17713.7, +39950.2,
   -31216.2) 1/m. Narrowing the strip raises BOTH L and Q here.
R3 against FD4 (1 % steps, four forward solves each, through cuDSS):
   dL/dr_out 6.04e-09, dQ/dr_out 1.21e-06, dL/dwidth 2.12e-09, dQ/dwidth
   1.13e-07; worst 1.21e-06. PASSES (<= 1e-4).

LEONTOVICH VALIDITY (``leontovich``; analytic, no gate). delta = 1.841 um at
2.45 GHz and 3.05e7 S/m: delta / t = 0.614 on the 3 um TopMetal2 and 0.921 on
the 2 um underpass and via -- NOT << 1. The exact 1-D internal impedance of a
slab driven from both faces, (Zs/2) coth((1+j) t / (2 delta)), against the two
independent sheets the model uses: resistance x 1.275 (M2) and x 1.855 (M1),
internal reactance x 0.537 / x 0.361. The sheet therefore UNDER-states the
metal resistance by roughly a fifth on the strip, so the absolute Q_diff is
optimistic; delta reaches t/3 on M2 only at 8.3 GHz. Across the sweep delta
runs from 4.08 um (0.5 GHz) to 1.02 um (8 GHz).

WALLS (``walls``; level 1, SuperLU on this Mac, run by ``--walls``). The
ladder track found the 10 W walls not converged for the paper geometry in
its protocol (vacuum, 120 um silicon, 10 MHz: 5 / 10 / 20 W moved L by
+3.09 % / +0.48 % at level 1). Measured here on this study's model, walls
and lid moved together: at the paper values 5 / 10 / 20 W give L_diff =
3160.67 / 3294.16 / 3321.26 pH, +4.22 % then +0.82 % (Q_diff -0.03 %), a
larger bias than the ladder's -- the thicker silicon puts the ground image
farther down and the fringing field wider. If every doubling kept the ratio
of these two steps, about 0.20 % more would come beyond 20 W (a
geometric-series estimate). The traced metric pins the walls in space while
theta moves (delta = 0 at the walls), so the wall-to-strip distance changes
across part C's design box, and so does the bias: 10 -> 20 W raises L_diff
by +0.66 % at the design point theta* (walls 317.8 um beyond r_out) and by
+1.72 % at the sweep point (260, 14, 27) um (258.0 um). The 10 W CPU solves
reproduce the GPU lanes' records of the same models to 1.2e-11 / 2.1e-12 (L,
nominal / theta*): SuperLU and cuDSS factor the same matrix. Stated, not
applied: the absolute L here is low by this much more, and carrying the
level-1 change to level 2 is an estimate (a 20 W level-1 solve is N =
125783, 442 s and 15.1 GB on this Mac; level 2 at 20 W does not fit it).

REMESH (``remesh``; level 1, CPU, ``--remesh``). Part C reaches every
theta by deforming the NOMINAL mesh with the traced metric, so at theta* the
22 um strip still has the nominal 2 cells, 27 % narrower. A FRESH level-1
mesh built at theta* by the same rule (base_dx 30 um, the spiral's
breakpoints there, the same physical walls and lid) is N = 188552 with 3
cells across the strip (one a 2.73 um sliver); it gives L_diff +0.52 % and
Q_diff +2.18 % against the deformed mesh (3786.11 against 3766.33 pH; 790 s,
21.5 GB). Not a same-resolution comparison -- the fresh mesh is finer across
the strip -- but it sizes what re-meshing would change at level 1: 8.5 %
of the level 1 -> 2 step at theta* on L (+6.21 %), 17.5 % of it on Q
(+12.43 %), i.e. inside the grid error, not on top of it.

ACCURACY (``accuracy``). Quoted from the ladder track (study P): on this
fixture family the FDFD's absolute L is LOW -- the small spiral's
extrapolated limit is 2.48-4.56 % below its Greenhouse referee, and the
paper geometry itself sat 13.71 % / 6.27 % below its referee at 2 / 4 cells
across W, this study's levels 1 / 2. The TRANSFER is approximate: the ladder
measured with vacuum dielectrics, a uniform current at 10 MHz and 120 um
silicon; this model has the dielectrics, a Leontovich sheet at 2.45 GHz and
190 um graded silicon. Against its own referee (the ground image at the
bottom of the 190 um silicon, 3782.07 pH) L_diff here is 12.90 % / 7.27 %
low at levels 1 / 2 -- not a pure grid bias either, since the FDFD carries
the dielectrics and the sheet's internal inductance and the referee does
not. On top of the grid, the 10 W walls are a low bias (WALLS). The
absolute L and Q carry all of this; derivatives are gated separately (R3).

RUNNING IT. The GPU work is done by ``validation/vessl/lane_r_rfic.py``
(lanes r0, r1; ``build_gpu_lanes.py``; the run ledger is
``validation/vessl/runs/r_run_ledger.json``). This file assembles the JSON and
figure from the harvested lane JSONs and recomputes the referee; the CPU
checks write their records into the JSON one solve at a time (``--walls
[N]`` solves the missing wall cases, ``--remesh`` the fresh mesh; an assembly
always carries them over)::

    .venv/bin/python validation/fdfd/rfic_spiral.py --walls 1   # repeat until none left
    .venv/bin/python validation/fdfd/rfic_spiral.py --remesh
    .venv/bin/python validation/fdfd/rfic_spiral.py [--from-json] [--no-figure]

Figure: (a) L and Q by level, (b) the sweep, (c) the substrate thickness
against the referee's ground image, (d) AD vs FD4, (e) the wall check,
(f) re-meshing, level step and walls at theta*.
"""
from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import math
import pathlib
import sys
import time
from typing import Any, Callable, Iterator, Sequence

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
JSON_PATH = HERE / "rfic_spiral.json"
PNG_PATH = HERE / "rfic_spiral.png"
REFEREE_PATH = REPO / "validation" / "crossval" / "comparators" / "spiral_greenhouse.py"
CONV_PATH = HERE / "spiral_convergence.py"
LADDER_JSON = HERE / "invariant_ladder.json"
RUNS = REPO / "validation" / "vessl" / "runs"
LEDGER = RUNS / "r_run_ledger.json"
PRIOR_GIT = "a9c35192118279ad56c2e383065ed7f5816a4fd7"

# ---------------------------------------------------------------------------
# the geometry: the paper's (r_out, W, S) as a SQUARE spiral (see the doc)

N_TURNS = 3
R_OUT, SPACING, WIDTH = 218e-6, 14e-6, 30e-6
THETA0 = (R_OUT, SPACING, WIDTH)
PARAMS = ("r_out", "spacing", "width")
LEAD = 1.5 * WIDTH
MARGIN = WIDTH

# stack (SG13G2-like, the numbers study H chose; see ``PRIOR_H`` for its
# documented deviations, which are kept)
T_OX_LOW = 6.04e-6
T_M1 = 2.0e-6
T_VIA = 2.0e-6
T_M2 = 3.0e-6
T_OX_HIGH = 1.5e-6
T_AIR = 60e-6
BASE_DZ = 60e-6
EPS_SI, EPS_OX = 11.9, 4.1
DZ_UNDER = 0.5 * T_M1 + T_VIA + 0.5 * T_M2

# ---------------------------------------------------------------------------
# LEGACY (study H) -- kept, values unchanged, because two other files import
# them: ``invariant_ladder.paper_referee`` (the referee at H's 120 um
# substrate, GROUND_H = 131.54 um) and ``validation/vessl/lane_j2_paper.py``
# (H's cell-defined ``build_level``). Study R does not use H's fixture.

T_SI = 120e-6                # H's substrate (the ladder's paper-geometry referee)
T_SI_ALT = 60e-6
T_SI_REAL = 190e-6
METAL_CELLS = 2
PAD_CELLS, PAD_CELLS_Z, PAD_RATIO = 4, 3, 1.5
FREQ_A = 1.0e8
SIGMA_VOL = 3.0e6
FD_STEP_REL = 0.01           # 1 % FD steps (far above the ~1e-9 LU noise floor)


def ground_height(t_si: float = T_SI) -> float:
    """PEC ground -> M2 centre line: the referee's image distance."""
    return t_si + T_OX_LOW + T_M1 + T_VIA + 0.5 * T_M2


GROUND_H = ground_height()
REF_SHIFT = 0.5 * WIDTH
BRIDGE_SPLIT = 0.5

# ---------------------------------------------------------------------------
# study R: the level-invariant fixture of ``rfx.fdfd.spiral`` (the options the
# ladder track added) with the real dielectrics, a graded substrate, lossy
# silicon and a Leontovich metal

T_SI_R = 190e-6              # chosen substrate thickness (graded, see graded_substrate)
T_SI_SENS = (120e-6, 190e-6, 300e-6)     # the measured thickness sensitivity (level 1)
T_SI_SENS_L2 = 120e-6        # the one alternative thickness also run at the resolved level
# neighbour ratio of the graded silicon cells: 2, not the in-plane mesher's 1.5,
# because of the 24 GB card -- cuDSS's plan of the resolved level (R0 lane) is
# 22.38 GB of factor at 1.5 (N = 823302; the solve died with ALLOC_FAILED on the
# 23.65 GB the card has free) and 20.51 GB at 2 (N = 772892)
SI_GRADE = 2.0
SI_GRADE_PROBE = 1.5         # the R0 probe's grading (level-1 comparison only)
PORT_GAP = 60e-6             # physical port gap (column bottom), = the ladder's paper fixture
SHORT_GAP = WIDTH            # physical short-standard post gap, = the ladder's
WALL_W = 10.0                # walls and lid 10 W from the DUT box (the ladder's rule)
METAL_CELLS_R = 1            # level-1 cells per metal slab (then subdivided with the level)
FREQ = 2.45e9
SIGMA_METAL = 3.05e7         # TopMetal2 as the task gives it (Leontovich sheet)
SIGMA_SI = 2.0               # p-substrate, S/m
SWEEP_GHZ = (0.5, 1.0, 2.0, 2.45, 3.5, 5.0, 6.5, 8.0)
LEVELS = (1, 2)              # 2 and 4 cells across W
RESOLVED = 2                 # the resolved level (4 cells across W)
FINER = 3                    # one level finer (6 cells across W)
FD_PARAMS = ("r_out", "width")   # FD4 checks at the resolved level (both L and Q)

RECIP_TOL = 1e-8
PASSIVITY_TOL = 1.0 + 1e-8
AD_FD_TOL = 1e-4
MU0 = 4e-7 * math.pi

L_TARGET = 4.000e-9          # the design target of part C (rfic_design.py)
Q_PAPER = 16.80              # context only


def fd4(f: Callable[[float], float], x0: float, d: float) -> float:
    return (-f(x0 + 2 * d) + 8 * f(x0 + d) - 8 * f(x0 - d) + f(x0 - 2 * d)) / (12 * d)


def skin_depth(freq: float, sigma: float) -> float:
    return float(np.sqrt(2.0 / (2.0 * np.pi * freq * MU0 * sigma)))


# ---------------------------------------------------------------------------
# 0. the review of study H (the previous rfic_spiral.py / .json, git PRIOR_GIT)

PRIOR_H: dict[str, Any] = {
    "source": f"git show {PRIOR_GIT[:8]}:validation/fdfd/rfic_spiral.json (and .py): built by "
              "an agent that died before reporting; reviewed here, not re-run",
    "fixture": "cell-defined (pad_cells = 4 graded wall cells, post one cell from each "
               "column, port gap one z cell, vertical grid frozen at base_dz = 60 um), "
               "t_si = 120 um uniform, SuperLU on a 38 GB Mac; base_dx = W and W/2 are the "
               "SAME grid (2 cells across W), N = 85785 / 83820",
    "R1": {"measured": {"W/1": -0.13218404790384464, "W/2": -0.13258911856264166},
           "claim": "absolute L at 100 MHz, sigma 3e6 S/m, vs the Greenhouse referee "
                    "(3560.37 pH), within 3 points of D2's W/2 gap",
           "recorded": True,
           "stands": False,
           "why": "invalid protocol at W = 30 um: sigma = 3e6 S/m at 100 MHz is a 29 um "
                  "skin depth, ~1 W, not the uniform-current regime it assumed (the ladder "
                  "later showed the crowding itself is small, +0.012 % at 0.97 W), and the "
                  "fixture was cell-defined with a never-refined vertical grid, which the "
                  "ladder's P6 identified as the cause of the old plateau. Both 'levels' "
                  "were one grid. Superseded by invariant_ladder.json 'paper' "
                  "(10 MHz / 2e6 S/m, level-invariant fixture: -13.71 % / -6.27 % at "
                  "2 / 4 cells across W)."},
    "R2": {"recorded": False,
           "measured": {"reciprocity_worst": 1.9704589366182672e-10,
                        "passivity_worst": 1.000000000107069,
                        "wall_rel_change_L_pad4_to_8": 0.010245359246944474,
                        "wall_steps_L": {"4->6": 0.009832471874017035,
                                         "6->8": 0.0004088671977056091}},
           "stands": "reciprocity and passivity: yes (12 solves); the gate as a whole: no",
           "why_failed": "NOT reciprocity or passivity -- both passed (1.97e-10, "
                         "1 + 1.07e-10). R2 carried a third sub-gate, side-wall "
                         "independence, which failed: pushing the cell-defined walls from "
                         "4 to 8 graded cells (431 -> 1356 um from the axis) moved L_diff "
                         "by +1.0245 % against a < 1 % criterion (steps +0.983 % then "
                         "+0.041 %): the pad-4 box every part used was 1.0 % low from its "
                         "walls alone. Study R's fixture puts the walls at a physical "
                         "10 W = 300 um from the DUT box (the ladder's rule)."},
    "R3": {"recorded": True, "stands": "yes, for H's own coarse model only",
           "measured": {"dL_dr_out": 1.4020244881791309e-08,
                        "dQ_dwidth": 5.7466826121904616e-08},
           "why": "AD vs FD4 of the same discrete model, 1 % steps; the check is sound but "
                  "the model (2 cells across W, cell-defined fixture) is superseded"},
    "R4": {"recorded": False, "stands": False,
           "why": "the JSON has no part-C record (R4 'complete': false). The docstring's "
                  "design numbers (theta* = 211.031 / 10.320 / 22.000 um, 3989.62 pH) "
                  "exist nowhere on disk and cannot be verified; they are discarded"},
    "R5": {"recorded": True, "stands": "as a historical cost record of the CPU path only",
           "measured": {"largest_n": 119689, "peak_footprint_gb": 12.444864749908447}},
    "B_2p45GHz": {"L_diff_W1": 3.043329885586503e-09, "Q_diff_W1": 20.64536530369604,
                  "stands": False,
                  "why": "2 cells across W, 120 um uniform silicon, walls 1 % close; "
                         "superseded by part A here"},
}


# ---------------------------------------------------------------------------
# 1. the referee (host numpy; study D2's construction rebound to this geometry)

def _load(path: pathlib.Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_referee():
    """The Greenhouse comparator (not a package; loaded by path)."""
    return _load(REFEREE_PATH, "spiral_greenhouse")


def bind_referee():
    """``spiral_convergence`` with its module-level fixture constants rebound
    to this geometry (its ``bridge_segments`` / ``referee_segments`` read them
    as globals); ``lead`` and the ground height are always passed explicitly."""
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
                shift: float | None = None, ground: float | None = GROUND_H):
    """Greenhouse terms of the strip + underpass (lead to the reference plane)."""
    segs = cv.referee_segments(sg, theta, convention, lead=LEAD, shift=shift)
    return sg.greenhouse_terms(segs, ground_height=ground)


def bridge_terms(sg, cv, theta: Sequence[float], shift: float | None = None,
                 ground: float | None = GROUND_H, split: float = BRIDGE_SPLIT):
    """Greenhouse terms of the short standard's bridge (D2's construction)."""
    return sg.greenhouse_terms(cv.bridge_segments(sg, theta, shift=shift, split=split),
                               ground_height=ground)


def referee_deembedded(sg, cv, theta: Sequence[float], shift: float | None = None,
                       ground: float | None = GROUND_H, split: float = BRIDGE_SPLIT,
                       convention: str = "area_exact") -> float:
    """``L(strip) - L(bridge)``: what the de-embedded ``L_diff`` measures."""
    return float(strip_terms(sg, cv, theta, convention, shift, ground).total
                 - bridge_terms(sg, cv, theta, shift, ground, split).total)


def referee_curve(thetas: Sequence[Sequence[float]] = (THETA0,),
                  t_sis: Sequence[float] = T_SI_SENS) -> dict[str, Any]:
    """The referee (vacuum, DC, ground image at the substrate bottom) at each
    substrate thickness -- the independent curve the FDFD's thickness
    sensitivity is read against (the silicon is non-magnetic and at 2 S/m its
    2.45 GHz skin depth is 7.2 mm, so to first order the substrate moves L
    through the ground image only)."""
    sg, cv = load_referee(), bind_referee()
    out: dict[str, Any] = {}
    for t in t_sis:
        out[f"{t * 1e6:g}"] = {"t_si": t, "ground_height": ground_height(t),
                               "deembedded": referee_deembedded(sg, cv, thetas[0],
                                                                ground=ground_height(t))}
    out["free_space"] = referee_deembedded(sg, cv, thetas[0], ground=None)
    return out


def referee_block() -> dict[str, Any]:
    """LEGACY (study H's name, kept for ``lane_j2_paper.py``): the primary
    referee at H's 120 um substrate -- strip, bridge and their difference."""
    sg, cv = load_referee(), bind_referee()
    st, br = strip_terms(sg, cv, THETA0), bridge_terms(sg, cv, THETA0)
    return {"strip": float(st.total), "bridge": float(br.total),
            "deembedded": float(st.total - br.total), "ground_height": GROUND_H}


def build_level(div: float = 1.0, eps: bool = True, pad: int = PAD_CELLS,
                pad_z: int = PAD_CELLS_Z, t_si: float = T_SI, base_dz: float = BASE_DZ,
                metal_cells: int = METAL_CELLS, margin: float = MARGIN):
    """LEGACY (study H's cell-defined fixture, kept for ``lane_j2_paper.py``)."""
    from rfx.fdfd import spiral as sm
    st = sm.SmallStack(t_si=t_si, t_ox_low=T_OX_LOW, t_m1=T_M1, t_via=T_VIA, t_m2=T_M2,
                       t_ox_high=T_OX_HIGH, t_air=T_AIR, base_dz=base_dz,
                       metal_cells=metal_cells, eps_si=EPS_SI if eps else 1.0,
                       eps_ox=EPS_OX if eps else 1.0)
    return sm.build_spiral(sm.SpiralSpec(
        n_turns=N_TURNS, r_out=R_OUT, spacing=SPACING, width=WIDTH, lead=LEAD,
        base_dx=WIDTH / div, margin=margin, stack=st, port_gap_cells=1, pad_cells=pad,
        pad_cells_z=pad_z, pad_ratio=PAD_RATIO))


def model_record(model) -> dict[str, Any]:
    """LEGACY grid record of study H's models (``lane_j2_paper.py``)."""
    x = np.asarray(model.x_nom)
    return {"shape": list(model.shape), "n_unknowns": int(model.n_unknowns),
            "base_dx": float(model.spec.dx), "dx_min": float(np.diff(x).min()),
            "cells_across_width": int(round(WIDTH / float(np.diff(x).min()))),
            "wall_x": [float(x[0]), float(x[-1])], "lid_z": float(np.asarray(model.z)[-1]),
            "build_seconds": float(model.build_seconds)}


# ---------------------------------------------------------------------------
# 2. the fixture

def silicon_lines(st, port_gap: float = PORT_GAP, ratio: float = SI_GRADE) -> np.ndarray:
    """Interior z lines of the GRADED silicon: from the oxide's first cell down
    to the port-gap line, a geometric grading of neighbour ratio <= ``ratio``
    whose last line is pinned to the port line (``rfx.fdfd.spiral.pad_to``, the
    same routine the invariant fixture pads its walls with). Below the port
    line the gap is one ``base_dz`` cell as before."""
    from rfx.fdfd import spiral as sm
    n_ox = max(1, int(np.ceil(st.t_ox_low / st.base_dz - 1e-9)))
    d_ox = st.t_ox_low / n_ox
    lines = sm.pad_to(np.array([st.t_si, st.t_si + d_ox]), port_gap, ratio, hi=False)
    return lines[1:-2]


@contextlib.contextmanager
def graded_substrate(port_gap: float = PORT_GAP, ratio: float = SI_GRADE) -> Iterator[None]:
    """Build the invariant fixture with a GRADED silicon (scoped).

    ``rfx.fdfd.spiral`` makes the silicon layer-uniform (``_stack_z_lines``:
    cells of at most ``base_dz`` between mandatory lines) and has no option
    for grading inside a layer; the study does not edit it. Instead the
    invariant fixture's own hook for mandatory z lines -- the ``extra``
    argument through which ``port_gap_m`` enters -- is fed the graded lines
    of :func:`silicon_lines` for the duration of one ``build_spiral`` call and
    restored afterwards (checked by the test). The lines are level-1 lines, so
    ``refine = m`` subdivides them like every other interval and the levels
    stay nested."""
    from rfx.fdfd import spiral as sm
    orig = sm._stack_z_lines

    def wrapped(st, stack, extra):
        return orig(st, stack, list(extra) + silicon_lines(st, port_gap, ratio).tolist())

    sm._stack_z_lines = wrapped
    try:
        yield
    finally:
        sm._stack_z_lines = orig


def stack_r(t_si: float = T_SI_R, eps: bool = True, base_dz: float = BASE_DZ,
            metal_cells: int = METAL_CELLS_R, t_air: float = T_AIR):
    from rfx.fdfd import spiral as sm
    return sm.SmallStack(t_si=t_si, t_ox_low=T_OX_LOW, t_m1=T_M1, t_via=T_VIA, t_m2=T_M2,
                         t_ox_high=T_OX_HIGH, t_air=t_air, base_dz=base_dz,
                         metal_cells=metal_cells,
                         eps_si=EPS_SI if eps else 1.0, eps_ox=EPS_OX if eps else 1.0)


GEOM = {"n_turns": N_TURNS, "r_out": R_OUT, "spacing": SPACING, "width": WIDTH,
        "lead": LEAD, "margin": MARGIN, "port_gap": PORT_GAP, "short_gap": SHORT_GAP,
        "t_air": T_AIR}


def spec_r(m: int, t_si: float = T_SI_R, wall_w: float = WALL_W, eps: bool = True,
           base_dz: float = BASE_DZ, geom: dict[str, Any] | None = None, z_refine: int = 0):
    """The level-invariant fixture at level ``m`` (walls and lid ``wall_w``
    widths from the DUT box, physical post and port gap, nested joint
    refinement; ``rfx.fdfd.spiral`` options only). ``geom`` overrides
    entries of :data:`GEOM` (only the CPU test model does); ``z_refine`` > 0
    refines z by that factor instead of ``m`` (the fixture's own option)."""
    from rfx.fdfd import spiral as sm
    g = dict(GEOM, **(geom or {}))
    st = stack_r(t_si, eps, base_dz, t_air=g["t_air"])
    wm = float(wall_w) * g["width"]
    return sm.SpiralSpec(n_turns=int(g["n_turns"]), r_out=g["r_out"], spacing=g["spacing"],
                         width=g["width"], lead=g["lead"], base_dx=g["width"],
                         margin=g["margin"], stack=st, wall_margin_m=wm,
                         lid_height_m=st.z_top + wm, short_gap_m=g["short_gap"],
                         port_gap_m=g["port_gap"], refine=int(m), z_refine=int(z_refine),
                         pad_ratio=PAD_RATIO)


def build_r(m: int, t_si: float = T_SI_R, grade: float | None = SI_GRADE,
            wall_w: float = WALL_W, eps: bool = True, base_dz: float = BASE_DZ,
            geom: dict[str, Any] | None = None, z_refine: int = 0):
    """The study's model at level ``m``; ``grade=None`` is the layer-uniform
    silicon of ``rfx.fdfd.spiral`` (the comparison), else graded."""
    from rfx.fdfd import spiral as sm
    spec = spec_r(m, t_si, wall_w, eps, base_dz, geom, z_refine)
    ctx = (graded_substrate(port_gap=spec.port_gap_m, ratio=grade) if grade
           else contextlib.nullcontext())
    with ctx:
        return sm.build_spiral(spec)


# the model the TESTS solve live on the CPU: the same code path (graded
# substrate hook, invariant walls / post / port gap, dielectrics, Leontovich
# metal, lossy silicon, three fixtures, de-embedding, traced metric) on a
# SMALL spiral (2 turns, r_out 60 um, W = S = 10 um) so that a three-fixture
# SuperLU solve is seconds; the paper-size level-1 model is N = 100126 and
# 339 s per three-fixture forward on this Mac (``walls_cpu`` nominal_w10).
# It is NOT a physics level: no accuracy claim
# is made on it, only the pipeline properties (reciprocity, passivity, AD vs
# FD4, the graded hook) are.
CHEAP = {"wall_w": 2.0, "base_dz": 20e-6, "t_si": 60e-6,
         "geom": {"n_turns": 2, "r_out": 60e-6, "spacing": 10e-6, "width": 10e-6,
                  "lead": 15e-6, "margin": 10e-6, "port_gap": 20e-6, "short_gap": 10e-6,
                  "t_air": 20e-6}}


def build_cheap():
    return build_r(1, **CHEAP)


def cells_across_width(model) -> int:
    """x cells inside the outer strip's right side, [r_out - W, r_out]."""
    x = np.asarray(model.x_nom)
    r_out, _, w = model.theta_nominal
    tol = 1e-9 * w
    return int(np.sum((x > r_out - w + tol) & (x < r_out - tol))) + 1


def grid_record(model) -> dict[str, Any]:
    x, yv, z = (np.asarray(v) for v in (model.x_nom, model.y_nom, model.z))
    st = model.spec.stack
    zsi = z[z <= st.t_si + 1e-12]
    return {"shape": list(model.shape), "n_unknowns": int(model.n_unknowns),
            "level": int(model.spec.refine), "z_refine": int(model.spec.z_refine or model.spec.refine),
            "cells_across_width": cells_across_width(model),
            "wall_x": [float(x[0]), float(x[-1])], "wall_y": [float(yv[0]), float(yv[-1])],
            "lid_z": float(z[-1]), "t_si": float(st.t_si),
            "dx_min": float(np.diff(x).min()), "dz_min": float(np.diff(z).min()),
            "n_z_cells": int(len(z) - 1), "n_si_cells": int(len(zsi) - 1),
            "dz_si": np.diff(zsi).tolist(),
            "port_gap_m": float(z[int(model.fixture.get("port_gap_index", 1))]),
            "build_seconds": float(model.build_seconds)}


# ---------------------------------------------------------------------------
# 3. solves

def s_stats(s) -> dict[str, float]:
    s = np.asarray(s)
    return {"reciprocity": float(abs(s[0, 1] - s[1, 0])),
            "max_singular_value": float(np.linalg.svd(s, compute_uv=False).max())}


def _c2(a) -> list:
    a = np.asarray(a)
    return np.stack([a.real, a.imag], axis=-1).tolist()


def s_block(s_raw, s_open, s_short, s_dut) -> dict[str, Any]:
    """The four S-matrices of one solve and their reciprocity / passivity."""
    mats = {"dut": s_raw, "open": s_open, "short": s_short, "deembedded": s_dut}
    st = {k: s_stats(v) for k, v in mats.items()}
    return {"S": {k: _c2(v) for k, v in mats.items()}, "s_stats": st,
            "reciprocity_worst": max(v["reciprocity"] for v in st.values()),
            "passivity_worst": max(v["max_singular_value"] for v in st.values())}


def forward(model, freq: float = FREQ, theta: Sequence[float] | None = None,
            sigma_metal: float = SIGMA_METAL, sigma_si: float = SIGMA_SI) -> dict[str, Any]:
    """One three-fixture forward solve and every metric."""
    import jax
    import jax.numpy as jnp

    from rfx.fdfd import spiral as sm
    th = model.theta_nominal if theta is None else tuple(float(v) for v in theta)
    t0 = time.time()
    r = sm.solve_spiral(model, freq, theta=jnp.asarray(th, dtype=jnp.float64),
                        sigma_metal=sigma_metal, sigma_si=sigma_si)
    jax.block_until_ready(r.L_diff)
    rec: dict[str, Any] = {
        "freq": float(freq), "theta": list(th), "sigma_metal": sigma_metal, "sigma_si": sigma_si,
        "L_diff": float(r.L_diff), "Q_diff": float(r.Q_diff), "L_se": float(r.L_se),
        "Q_se": float(r.Q_se), "L_z11_open": float(r.L_z11_open), "L_raw": float(r.L_raw),
        "Z_dut": _c2(r.z_dut), "seconds": time.time() - t0}
    rec.update(s_block(r.s_raw, r.s_open, r.s_short, r.s_dut))
    return rec


def metrics_fn(model, freq: float = FREQ, sigma_metal: float = SIGMA_METAL,
               sigma_si: float = SIGMA_SI):
    """``theta -> ((L_diff, Q_diff), S-matrices)`` (traced; for ``jax.vjp``)."""
    import jax.numpy as jnp

    from rfx.fdfd import spiral as sm

    def f(theta):
        r = sm.solve_spiral(model, freq, theta=theta, sigma_metal=sigma_metal, sigma_si=sigma_si)
        return (jnp.real(r.L_diff), jnp.real(r.Q_diff)), (r.s_raw, r.s_open, r.s_short, r.s_dut)

    return f


def metrics_with_grads(model, freq: float = FREQ, theta: Sequence[float] | None = None,
                       **kw) -> dict[str, Any]:
    """L_diff, Q_diff and both gradients in theta: ONE forward (three fixture
    factorisations) and two reverse passes (``jax.vjp`` with the cotangents
    (1, 0) and (0, 1)); the S-matrices of the forward solve come back as aux."""
    import jax
    import jax.numpy as jnp
    th = jnp.asarray(model.theta_nominal if theta is None else theta, dtype=jnp.float64)
    t0 = time.time()
    (lv, qv), vjp_fn, aux = jax.vjp(metrics_fn(model, freq, **kw), th, has_aux=True)
    jax.block_until_ready(lv)
    t1 = time.time()
    one, zero = jnp.asarray(1.0), jnp.asarray(0.0)
    (gl,) = vjp_fn((one, zero))
    jax.block_until_ready(gl)
    t2 = time.time()
    (gq,) = vjp_fn((zero, one))
    jax.block_until_ready(gq)
    t3 = time.time()
    rec: dict[str, Any] = {"freq": float(freq), "theta": [float(v) for v in th],
                           "L_diff": float(lv), "Q_diff": float(qv),
                           "dL": [float(v) for v in gl], "dQ": [float(v) for v in gq],
                           "params": list(PARAMS), "units": {"dL": "H/m", "dQ": "1/m"},
                           "seconds": {"forward": t1 - t0, "reverse_L": t2 - t1,
                                       "reverse_Q": t3 - t2, "total": t3 - t0}}
    rec.update(s_block(*aux))
    return rec


def fd4_param(model, k: int, theta0: Sequence[float] | None = None, freq: float = FREQ,
              step_rel: float = FD_STEP_REL, log: Callable[[str], None] | None = None,
              **kw) -> dict[str, Any]:
    """4th-order central FD of L_diff AND Q_diff in parameter ``k``: four
    forward solves at +-h, +-2h (``h = step_rel * theta0[k]``), each recorded
    with its S-matrices."""
    from rfx.fdfd import spiral as sm
    th0 = list(model.theta_nominal if theta0 is None else theta0)
    h = step_rel * th0[k]
    pts: dict[str, Any] = {}
    for m in (-2, -1, 1, 2):
        t = list(th0)
        t[k] = th0[k] + m * h
        sm.check_feasible(model, np.asarray(t))
        pts[str(m)] = forward(model, freq, t, **kw)
        if log:
            log(f"    FD {PARAMS[k]} {m:+d}h: L={pts[str(m)]['L_diff'] * 1e12:.4f} pH "
                f"Q={pts[str(m)]['Q_diff']:.5f} ({pts[str(m)]['seconds']:.0f} s)")

    def stencil(key: str) -> float:
        v = {m: pts[str(m)][key] for m in (-2, -1, 1, 2)}
        return (-v[2] + 8 * v[1] - 8 * v[-1] + v[-2]) / (12 * h)

    return {"param": PARAMS[k], "k": k, "theta0": th0, "step": h, "step_rel": step_rel,
            "fd_order": 4, "points": pts, "fd_L": stencil("L_diff"), "fd_Q": stencil("Q_diff")}


# ---------------------------------------------------------------------------
# 4. the Leontovich statement (analytic; independent of the FDFD)

def slab_ratio(t: float, delta: float) -> dict[str, float]:
    """A metal slab of thickness ``t`` driven from both faces: its exact 1-D
    internal impedance is ``(Zs / 2) coth(gamma t / 2)`` with ``gamma = (1 +
    j) / delta``, while two independent Leontovich sheets give ``Zs / 2``. The
    ratio ``coth(gamma t / 2)`` therefore says how far the sheet is off in
    resistance (real part of ``(1 + j) coth``) and internal reactance
    (imaginary part) at this delta / t. It tends to 1 for t >> delta and to
    ``2 delta / t`` (the DC resistance) for t << delta."""
    g = (1 + 1j) * t / (2 * delta)
    c = complex((1 + 1j) * np.cosh(g) / np.sinh(g))
    return {"t": t, "delta_over_t": delta / t, "R_exact_over_R_sheet": c.real,
            "X_exact_over_X_sheet": c.imag, "R_dc_over_R_sheet": 2 * delta / t}


def leontovich_block(freq: float = FREQ, sigma: float = SIGMA_METAL) -> dict[str, Any]:
    d = skin_depth(freq, sigma)
    return {"freq": freq, "sigma_metal": sigma, "skin_depth": d,
            "delta_over_t": {"M2": d / T_M2, "M1": d / T_M1, "VIA": d / T_VIA},
            "leontovich_validity": d / min(T_M1, T_VIA, T_M2),
            "slab_1d": {"M2": slab_ratio(T_M2, d), "M1": slab_ratio(T_M1, d)},
            "sweep_skin_depth": {f"{f:g}": skin_depth(f * 1e9, sigma) for f in SWEEP_GHZ},
            "frequency_where_delta_is_t_over_3_on_M2":
                2.0 / (2 * np.pi * MU0 * sigma * (T_M2 / 3) ** 2)}


# ---------------------------------------------------------------------------
# 4b. the wall check (CPU, SuperLU, level 1; run here, not on the GPU)
#
# The ladder track measured that 10 W walls are NOT converged for the paper
# geometry in its protocol (vacuum dielectrics, 120 um silicon, 10 MHz):
# 5 / 10 / 20 W moved L by +3.09 % then +0.48 % at level 1. This study's
# silicon is 190 um (the ground image is farther, the fringing wider), and
# the traced metric pins the walls in space while theta moves, so the
# wall-to-DUT distance changes across the design box. Both are measured
# here: the same level-1 model with walls and lid at 5 / 10 / 20 W, at the
# paper values, at the design's theta* and at the sweep point with the
# largest r_out, each point reached by the traced metric exactly as the
# design and the sweep reach it. Level 1 because a 20 W level-2 model does
# not fit this Mac (a 20 W level-1 solve is N = 125783, 28 GB peak, 444 s
# single-threaded); the wall effect is a far-field one, carried to level 2
# as an estimate (``walls`` says so).

WALL_CHECK_LEVEL = 1
WALL_CHECK_CASES = (("nominal", 5.0), ("nominal", 10.0), ("nominal", 20.0),
                    ("design", 10.0), ("design", 20.0),
                    ("r_out_hi", 10.0), ("r_out_hi", 20.0))
THETA_R_OUT_HI = (260e-6, 14e-6, 27e-6)     # a sweep point: r_out at the box's top
DESIGN_JSON = HERE / "rfic_design.json"


def wall_check_points() -> dict[str, tuple[float, float, float]]:
    """The three thetas of the wall check; theta* is read from the design's
    JSON (``gates.R4.theta``, the returned L-BFGS-B point)."""
    pts: dict[str, tuple[float, float, float]] = {"nominal": THETA0,
                                                  "r_out_hi": THETA_R_OUT_HI}
    if DESIGN_JSON.exists():
        th = json.loads(DESIGN_JSON.read_text()).get("gates", {}).get("R4", {}).get("theta")
        if th:
            pts["design"] = (float(th[0]), float(th[1]), float(th[2]))
    return pts


def wall_case_key(point: str, wall_w: float) -> str:
    return f"{point}_w{wall_w:g}"


def run_wall_check(max_cases: int | None = None, log: Callable[[str], None] = print) -> int:
    """Solve the missing cases of :data:`WALL_CHECK_CASES` on the CPU
    (SuperLU, factor cache off so one factor is held at a time) and write
    each record into ``rfic_spiral.json`` ``walls_cpu`` as soon as it
    exists (a killed run loses at most the case in flight). Returns the
    number of cases solved."""
    import resource

    import jax

    from rfx.fdfd import spiral as sm
    from rfx.fdfd.linear_solve import factor_cache_size
    pts = wall_check_points()
    done = 0
    models: dict[float, Any] = {}
    old = factor_cache_size()
    factor_cache_size(0)
    try:
        for point, w in WALL_CHECK_CASES:
            if max_cases is not None and done >= max_cases:
                break
            study = json.loads(JSON_PATH.read_text()) if JSON_PATH.exists() else {}
            blk = study.setdefault("walls_cpu", {"cases": {}})
            key = wall_case_key(point, w)
            if key in blk["cases"] or point not in pts:
                continue
            if w not in models:
                models = {w: build_r(WALL_CHECK_LEVEL, wall_w=w)}
            m = models[w]
            th = pts[point]
            sm.check_feasible(m, np.asarray(th))
            rec = forward(m, theta=th)
            g = grid_record(m)
            rec.update(point=point, wall_w=w, level=WALL_CHECK_LEVEL, grid=g,
                       wall_x_minus_r_out=g["wall_x"][1] - th[0],
                       backend="superlu (CPU)", jax=jax.__version__,
                       max_rss_gb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9)
            study = json.loads(JSON_PATH.read_text()) if JSON_PATH.exists() else {}
            study.setdefault("walls_cpu", {"cases": {}})["cases"][key] = rec
            JSON_PATH.write_text(json.dumps(study, indent=1, default=float) + "\n")
            done += 1
            log(f"  walls {key}: N = {g['n_unknowns']}  L = {rec['L_diff'] * 1e12:.4f} pH  "
                f"Q = {rec['Q_diff']:.5f}  ({rec['seconds']:.0f} s)")
    finally:
        factor_cache_size(old)
    return done


def build_fresh(theta: Sequence[float], like) -> Any:
    """A FRESH level-1 mesh built at ``theta`` by the same rule as the
    nominal one (``spec_r``: base_dx 30 um = the nominal W, the spiral's
    breakpoints at theta, graded silicon, physical post and port gap) with
    the walls and lid at the physical positions of ``like`` (the nominal
    10 W model the design deforms), so that only the mesh -- not the box --
    differs from the deformed model at the same theta."""
    import dataclasses

    from rfx.fdfd import spiral as sm
    g = dict(GEOM, r_out=float(theta[0]), spacing=float(theta[1]), width=float(theta[2]))
    spec = spec_r(WALL_CHECK_LEVEL, geom=g)
    spec = dataclasses.replace(spec, base_dx=WIDTH,
                               wall_margin_m=float(np.asarray(like.x_nom)[-1]) - float(theta[0]),
                               lid_height_m=float(np.asarray(like.z)[-1]))
    with graded_substrate(port_gap=spec.port_gap_m):
        return sm.build_spiral(spec)


def run_remesh_check(log: Callable[[str], None] = print) -> int:
    """The fresh level-1 mesh at theta* (:func:`build_fresh`), solved on the
    CPU into ``rfic_spiral.json`` ``remesh_cpu``; compared (``remesh``)
    with the deformed nominal mesh at the same theta (``walls_cpu``
    design_w10) -- the remeshing sensitivity of the design point."""
    import resource

    import jax

    from rfx.fdfd.linear_solve import factor_cache_size
    pts = wall_check_points()
    study = json.loads(JSON_PATH.read_text()) if JSON_PATH.exists() else {}
    if "design" not in pts or (study.get("remesh_cpu") or {}).get("fresh_design_w10"):
        return 0
    old = factor_cache_size()
    factor_cache_size(0)
    try:
        like = build_r(WALL_CHECK_LEVEL)
        m = build_fresh(pts["design"], like)
        rec = forward(m)
    finally:
        factor_cache_size(old)
    g = grid_record(m)
    rec.update(point="design", mesh="fresh at theta*", level=WALL_CHECK_LEVEL, grid=g,
               wall_x_minus_r_out=g["wall_x"][1] - pts["design"][0], backend="superlu (CPU)",
               jax=jax.__version__,
               max_rss_gb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9)
    study = json.loads(JSON_PATH.read_text()) if JSON_PATH.exists() else {}
    study.setdefault("remesh_cpu", {})["fresh_design_w10"] = rec
    JSON_PATH.write_text(json.dumps(study, indent=1, default=float) + "\n")
    log(f"  remesh fresh_design_w10: N = {g['n_unknowns']}  L = {rec['L_diff'] * 1e12:.4f} pH  "
        f"Q = {rec['Q_diff']:.5f}  ({rec['seconds']:.0f} s)")
    return 1


def remesh_block(fresh: dict[str, Any] | None, deformed: dict[str, Any] | None,
                 step12: dict[str, Any] | None) -> dict[str, Any]:
    """Fresh mesh at theta* against the deformed nominal mesh at theta*
    (both level 1, 10 W walls at the same physical positions, CPU)."""
    if not fresh or not deformed:
        return {}
    out = {"what": "the design point theta* on a FRESH level-1 mesh built there (same meshing "
                   "rule, same physical walls and lid) against the nominal level-1 mesh "
                   "DEFORMED to theta* by the traced metric (how the design, the sweep and "
                   "the finer grids reach every theta)",
           "fresh": {k: fresh[k] for k in ("L_diff", "Q_diff")}
           | {k: fresh["grid"][k] for k in ("n_unknowns", "cells_across_width", "dx_min",
                                            "shape", "wall_x", "wall_y", "lid_z")},
           "deformed": {k: deformed[k] for k in ("L_diff", "Q_diff")}
           | {k: deformed["grid"][k] for k in ("n_unknowns", "shape", "wall_x", "wall_y",
                                               "lid_z")},
           "dL_rel_fresh_vs_deformed": fresh["L_diff"] / deformed["L_diff"] - 1.0,
           "dQ_rel_fresh_vs_deformed": fresh["Q_diff"] / deformed["Q_diff"] - 1.0,
           "deformed_cells_across_width": 2,
           "reading": "not a same-resolution comparison: the mesher puts 3 cells (one a "
                      "sliver) across the 22 um strip of the fresh mesh, where the deformed "
                      "one keeps the nominal 2; the difference is what re-meshing at theta* "
                      "would change, of the order of a grid step, not a pure metric error"}
    if step12:
        out["level1_to_2_at_theta_star"] = step12
        out["fraction_of_level_step_L"] = out["dL_rel_fresh_vs_deformed"] / step12["dL_rel"]
        out["fraction_of_level_step_Q"] = out["dQ_rel_fresh_vs_deformed"] / step12["dQ_rel"]
    return out


def walls_block(cases: dict[str, Any], lv1: dict[str, Any] | None,
                opt_m1: dict[str, Any] | None) -> dict[str, Any]:
    """The wall check, derived: per point the relative change of L_diff and
    Q_diff from 10 W to 20 W (and 5 -> 10 W at the paper values), the
    wall-to-strip distance, and the cross-backend check of the 10 W CPU
    solves against the GPU lanes (r1 level 1, r4 opt_m1_w10)."""
    if not cases:
        return {}
    out: dict[str, Any] = {
        "what": "level-1 model with walls and lid at 5 / 10 / 20 W (x, y walls and lid "
                "moved together; the chosen fixture is 10 W), each point reached by the "
                "traced metric from the nominal mesh exactly as the design and the sweep "
                "reach it (walls pinned in space), solved on the CPU (SuperLU)",
        "level": WALL_CHECK_LEVEL, "points": {k: list(v) for k, v in wall_check_points().items()},
        "per_point": {}}
    by: dict[str, dict[float, dict[str, Any]]] = {}
    for rec in cases.values():
        by.setdefault(rec["point"], {})[float(rec["wall_w"])] = rec
    for p, rows in by.items():
        blk: dict[str, Any] = {
            "L_diff": {f"{w:g}": r["L_diff"] for w, r in sorted(rows.items())},
            "Q_diff": {f"{w:g}": r["Q_diff"] for w, r in sorted(rows.items())},
            "n_unknowns": {f"{w:g}": r["grid"]["n_unknowns"] for w, r in sorted(rows.items())},
            "wall_x_minus_r_out": {f"{w:g}": r["wall_x_minus_r_out"]
                                   for w, r in sorted(rows.items())}}
        for a, b in ((10.0, 20.0), (5.0, 10.0)):
            if a in rows and b in rows:
                blk[f"dL_rel_{a:g}_to_{b:g}W"] = rows[b]["L_diff"] / rows[a]["L_diff"] - 1.0
                blk[f"dQ_rel_{a:g}_to_{b:g}W"] = rows[b]["Q_diff"] / rows[a]["Q_diff"] - 1.0
        out["per_point"][p] = blk
    pp = out["per_point"]
    nm = pp.get("nominal", {})
    if "dL_rel_5_to_10W" in nm and "dL_rel_10_to_20W" in nm:
        r = nm["dL_rel_10_to_20W"] / nm["dL_rel_5_to_10W"]
        nm["doubling_ratio"] = r
        nm["tail_beyond_20W_geometric"] = nm["dL_rel_10_to_20W"] * r / (1.0 - r)
        nm["tail_what"] = ("if every further doubling of the wall distance shrinks the change "
                           "by the same ratio as 5 -> 10 -> 20 W, the L still to come beyond "
                           "20 W (a geometric-series estimate, not a solve)")
    if all(p in pp and "dL_rel_10_to_20W" in pp[p] for p in ("nominal", "design", "r_out_hi")):
        v = [pp[p]["dL_rel_10_to_20W"] for p in ("design", "nominal", "r_out_hi")]
        out["dL_rel_10_to_20W_range_over_box"] = [min(v), max(v)]
        out["design_minus_nominal_points"] = (pp["design"]["dL_rel_10_to_20W"]
                                              - pp["nominal"]["dL_rel_10_to_20W"])
    if LADDER_JSON.exists():
        wl = json.loads(LADDER_JSON.read_text()).get("paper", {}).get("walls_level1", {})
        if all(k in wl for k in ("5", "10", "20")):
            out["ladder_paper_protocol"] = {
                "source": "invariant_ladder.json paper.walls_level1 (vacuum dielectrics, "
                          "120 um silicon, 10 MHz, level 1)",
                "L": wl, "dL_rel_5_to_10W": wl["10"] / wl["5"] - 1.0,
                "dL_rel_10_to_20W": wl["20"] / wl["10"] - 1.0}
    xb: dict[str, Any] = {}
    nom10, des10 = by.get("nominal", {}).get(10.0), by.get("design", {}).get(10.0)
    if nom10 and lv1 and "L_diff" in lv1:
        xb["nominal_level1"] = {"cpu": nom10["L_diff"], "gpu_r1": lv1["L_diff"],
                                "rel_L": nom10["L_diff"] / lv1["L_diff"] - 1.0,
                                "rel_Q": nom10["Q_diff"] / lv1["Q_diff"] - 1.0}
    if des10 and opt_m1 and "L_diff" in opt_m1:
        xb["design_level1"] = {"cpu": des10["L_diff"], "gpu_r4": opt_m1["L_diff"],
                               "rel_L": des10["L_diff"] / opt_m1["L_diff"] - 1.0,
                               "rel_Q": des10["Q_diff"] / opt_m1["Q_diff"] - 1.0}
    out["cross_backend_10W"] = xb
    out["reading"] = ("10 W walls are a bias of the fixture, not a converged boundary: moving "
                      "them to 20 W raises L_diff at level 1 by the per-point dL_rel_10_to_20W; "
                      "the traced metric keeps the walls where the nominal mesh put them, so "
                      "the bias differs across the design box (smaller r_out = farther walls). "
                      "Stated, not applied; carried to level 2 as an estimate (the walls act "
                      "on the far field, which level 1 already resolves)")
    return out


# ---------------------------------------------------------------------------
# 5. assembly (from the harvested GPU lane JSONs)

def load_ledger() -> list[dict[str, Any]]:
    return json.loads(LEDGER.read_text()) if LEDGER.exists() else []


def lane_json(lane: str, fname: str) -> tuple[pathlib.Path | None, dict[str, Any]]:
    """The lane JSON of the ledger's USED run of ``lane`` (``use: true``)."""
    for row in load_ledger():
        if row.get("lane") == lane and row.get("use"):
            p = REPO / row["artifacts"] / fname
            if p.exists():
                return p, json.loads(p.read_text())
    return None, {}


def srf_fit(freqs_hz: Sequence[float], ls: Sequence[float]) -> dict[str, Any]:
    """Least squares of ``1/L = (1 - (f/f0)^2) / L0`` (linear in f^2) over
    the sweep points: the lumped-LC self-resonance ``f0`` the L(f) rise implies."""
    # in GHz and 1/nH: the two columns are then O(1) and O(10-100), so the
    # least squares is well conditioned (in Hz and 1/H they differ by 1e19)
    f2 = (np.asarray(freqs_hz, dtype=float) / 1e9) ** 2
    y = 1e-9 / np.asarray(ls, dtype=float)
    a = np.vstack([np.ones_like(f2), f2]).T
    (c0, c1), *_ = np.linalg.lstsq(a, y, rcond=None)
    l0 = 1e-9 / c0
    f0 = float(np.sqrt(-c0 / c1)) * 1e9 if c1 < 0 else float("nan")
    resid = y - a @ np.array([c0, c1])
    return {"L0": float(l0), "f_self_resonance": f0,
            "max_rel_residual": float(np.max(np.abs(resid / y)))}


def collect_s(prefix: str, obj: Any, out: dict[str, dict[str, float]]) -> None:
    """Every record carrying ``s_stats`` under ``obj``, flattened by path."""
    if isinstance(obj, dict):
        if "reused" in obj:           # the same solve recorded under a second key
            return
        if "s_stats" in obj and "reciprocity_worst" in obj:
            out[prefix] = {"reciprocity": obj["reciprocity_worst"],
                           "passivity": obj["passivity_worst"],
                           "reciprocity_solved": max(obj["s_stats"][k]["reciprocity"]
                                                     for k in ("dut", "open", "short")),
                           "passivity_solved": max(obj["s_stats"][k]["max_singular_value"]
                                                   for k in ("dut", "open", "short"))}
        for k, v in obj.items():
            if k in ("S", "s_stats"):
                continue
            collect_s(f"{prefix}.{k}" if prefix else k, v, out)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            collect_s(f"{prefix}[{i}]", v, out)


def r2_gate(solves: dict[str, dict[str, float]]) -> dict[str, Any]:
    rw = max((v["reciprocity"] for v in solves.values()), default=None)
    pw = max((v["passivity"] for v in solves.values()), default=None)
    return {"n_solves": len(solves), "n_s_matrices": 4 * len(solves),
            "reciprocity_worst": rw, "passivity_worst": pw,
            "reciprocity_worst_solved_only": max((v["reciprocity_solved"] for v in solves.values()),
                                                 default=None),
            "passivity_worst_solved_only": max((v["passivity_solved"] for v in solves.values()),
                                               default=None),
            "reciprocity_tolerance": RECIP_TOL, "passivity_tolerance": PASSIVITY_TOL,
            "what": "every S-matrix of every solve: the three SOLVED fixtures (dut, open, "
                    "short) and the de-embedded one; |S12 - S21| and the largest singular value",
            "per_solve": solves,
            "passed": bool(solves) and rw is not None and rw <= RECIP_TOL
            and pw is not None and pw <= PASSIVITY_TOL}


def accuracy_statement(study: dict[str, Any] | None = None) -> dict[str, Any]:
    """The ladder track's absolute-L bias, quoted (not re-derived), with what
    limits its transfer to this model; given ``study``, also this study's own
    numbers: the level-1/2 L_diff against the referee at 190 um and the
    measured wall bias (``walls``)."""
    if not LADDER_JSON.exists():
        return {}
    il = json.loads(LADDER_JSON.read_text())
    p3 = il["gates"]["P3"]
    pa = il.get("paper", {})
    wl = pa.get("walls_level1", {})
    out: dict[str, Any] = {
        "source": "validation/fdfd/invariant_ladder.json (study P)",
        "small_spiral_extrapolated_rel_range": p3.get("rel_range", p3.get("measured")),
        "small_spiral_P3": {k: p3[k] for k in p3 if k in ("rel_range", "measured", "passed",
                                                           "referee", "range")},
        "paper_geometry_level_gaps": {k: v["gap"] for k, v in pa.get("levels", {}).items()},
        "paper_geometry_two_point_rel_range": pa.get("rel_range"),
        "paper_geometry_protocol": "vacuum dielectrics, 10 MHz, uniform-current 2e6 S/m, "
                                   "t_si 120 um, the same invariant fixture as here",
        "paper_geometry_walls_level1": wl,
        "paper_geometry_walls_dL_rel_10_to_20W": (wl["20"] / wl["10"] - 1.0
                                                  if "10" in wl and "20" in wl else None),
        "transfer": "APPROXIMATE: the ladder's gaps were measured with vacuum dielectrics, a "
                    "uniform current at 10 MHz and 120 um silicon; this model has the "
                    "dielectrics, a Leontovich sheet at 2.45 GHz and 190 um graded silicon, so "
                    "the ladder's -6.27 % at 4 cells across W is a guide to the grid bias here, "
                    "not a measurement of it",
        "reading": "the FDFD's absolute L on this fixture family is LOW: at 4 cells across W "
                   "(this study's resolved level) the paper geometry sat 6.27 % below its "
                   "Greenhouse referee in the ladder's quasi-static protocol, and the small "
                   "spiral's extrapolated limit is 2.48-4.56 % below its referee. The 10 W "
                   "walls add a further low bias (the ladder: +0.48 % from 10 to 20 W at "
                   "level 1 for the paper geometry; this study's own is in ``walls``). "
                   "Absolute L and Q here carry these biases; derivatives are gated "
                   "separately (R3)."}
    if study:
        ref = (study.get("referee") or {}).get(f"{T_SI_R * 1e6:g}", {}).get("deembedded")
        lv = study.get("levels", {})
        if ref:
            out["own_referee_190um"] = {
                "referee_deembedded": ref,
                "gap_by_level": {k: v["L_diff"] / ref - 1.0 for k, v in lv.items()
                                 if isinstance(v, dict) and "L_diff" in v},
                "what": "L_diff at 2.45 GHz against the Greenhouse referee (vacuum, "
                        "uniform current, ground image at the bottom of 190 um silicon): NOT "
                        "a pure grid bias -- the FDFD carries the dielectrics and the "
                        "Leontovich sheet's internal inductance, which the referee does not"}
        w = (study.get("walls") or {}).get("per_point", {}).get("nominal", {})
        if "dL_rel_10_to_20W" in w:
            out["own_walls_dL_rel_10_to_20W_level1"] = w["dL_rel_10_to_20W"]
    return out


def assemble(reuse: dict[str, Any] | None = None) -> dict[str, Any]:
    t0 = time.time()
    study: dict[str, Any] = {
        "what": "study R parts A and B: the paper-scale square spiral at 2.45 GHz on the "
                "level-invariant fixture with a graded silicon, Leontovich metal and lossy "
                "substrate, through cuDSS on the GPU",
        "prior_H_review": PRIOR_H,
        "fixture": fixture_record(),
        "leontovich": leontovich_block(),
    }
    ref = (reuse or {}).get("referee") or referee_curve()
    study["referee"] = ref
    p0, r0 = lane_json("r0", "r0_probe.json")
    p1, r1 = lane_json("r1", "r1_study.json")
    study["sources"] = {"r0": str(p0.relative_to(REPO)) if p0 else None,
                        "r1": str(p1.relative_to(REPO)) if p1 else None}
    study["runs"] = [row for row in load_ledger() if row.get("lane", "").startswith("r")]
    plans = {k: v for k, v in dict(r0.get("plans", {}), **r1.get("plans", {})).items()
             if isinstance(v, dict) and "grid" in v}
    study["plans"] = {k: {kk: v.get(kk) for kk in ("n", "permanent_device_memory_gb",
                                                   "peak_device_memory_gb",
                                                   "hybrid_min_device_memory_gb")}
                      | {"n_unknowns": v["grid"]["n_unknowns"], "shape": v["grid"]["shape"]}
                      for k, v in plans.items()}
    lv0 = r0.get("levels", {})
    study["probe_r0"] = {
        "what": f"the R0 probe, silicon graded at ratio {SI_GRADE_PROBE} (level 1 only: its "
                "level-2 solve died with ALLOC_FAILED, see plans)",
        "tsi_level1": r0.get("tsi_level1", {}),
        "uniform_level1": (r0.get("uniform_vs_graded") or {}).get("uniform", {}),
        "level2_error": lv0.get("error", "")[-200:] if isinstance(lv0.get("error"), str) else None}
    study["levels"] = {k: v for k, v in r1.get("levels", {}).items() if isinstance(v, dict)}
    study["records"] = {"tsi_level1": r1.get("tsi_level1", {}),
                        "uniform_level1": (r1.get("uniform_vs_graded") or {}).get("uniform", {}),
                        "tsi_resolved": r1.get("tsi_resolved", {}),
                        "sweep": r1.get("sweep", {})}
    study["grading"] = grading_block(study)
    study["tsi_sensitivity"] = tsi_block(r1, ref)
    study["sweep"] = sweep_block(r1)
    study["gradients"] = r1.get("grad", {})
    study["fd"] = r1.get("fd", {})
    # the CPU wall check (measured by ``--walls`` into this JSON; always carried over)
    disk = json.loads(JSON_PATH.read_text()) if JSON_PATH.exists() else {}
    study["walls_cpu"] = (disk.get("walls_cpu") or (reuse or {}).get("walls_cpu")
                          or {"cases": {}})
    _, r4 = lane_json("r4", "r4_fine.json")
    study["walls"] = walls_block(study["walls_cpu"].get("cases", {}), study["levels"].get("1"),
                                 ((r4.get("fine") or {}).get("cases") or {}).get("opt_m1_w10"))
    study["remesh_cpu"] = disk.get("remesh_cpu") or (reuse or {}).get("remesh_cpu") or {}
    dj = json.loads(DESIGN_JSON.read_text()) if DESIGN_JSON.exists() else {}
    study["remesh"] = remesh_block(
        study["remesh_cpu"].get("fresh_design_w10"),
        study["walls_cpu"].get("cases", {}).get(wall_case_key("design", WALL_W)),
        (dj.get("gates", {}).get("R7") or {}).get("level1_to_2_at_optimum"))
    study["accuracy"] = accuracy_statement(study)
    env = r1.get("env", {})
    study["env"] = {k: env.get(k) for k in ("python", "jax", "scipy", "numpy", "nvidia_smi",
                                           "nvmath_version")}
    study["gates"] = evaluate_gates(study)
    study["seconds_assembly"] = time.time() - t0
    return study


def grading_block(study: dict[str, Any]) -> dict[str, Any]:
    """The substrate MESH at the chosen 190 um, level 1: layer-uniform silicon
    (``rfx.fdfd.spiral``'s own) vs graded at ratio 1.5 (R0) and 2 (the study)."""
    k = f"{T_SI_R * 1e6:g}"
    rows = {"uniform": study["records"].get("uniform_level1", {}),
            f"graded_{SI_GRADE_PROBE:g}": study["probe_r0"]["tsi_level1"].get(k, {}),
            f"graded_{SI_GRADE:g}": study["records"]["tsi_level1"].get(k, {})}
    out: dict[str, Any] = {"level": 1, "t_si": T_SI_R, "chosen": f"graded_{SI_GRADE:g}"}
    base = rows[f"graded_{SI_GRADE:g}"]
    for name, r in rows.items():
        if "L_diff" not in r:
            continue
        out[name] = {"L_diff": r["L_diff"], "Q_diff": r["Q_diff"],
                     "n_si_cells": r.get("grid", {}).get("n_si_cells"),
                     "n_unknowns": r.get("grid", {}).get("n_unknowns"),
                     "dz_si": r.get("grid", {}).get("dz_si")}
        if "L_diff" in base:
            out[name]["dL_rel_to_chosen"] = r["L_diff"] / base["L_diff"] - 1.0
            out[name]["dQ_rel_to_chosen"] = r["Q_diff"] / base["Q_diff"] - 1.0
    return out


def tsi_block(r1: dict[str, Any], ref: dict[str, Any]) -> dict[str, Any]:
    """Substrate-thickness sensitivity: the FDFD at level 1 (three
    thicknesses) and at the resolved level (two), against the referee's
    ground-image curve on the same thicknesses."""
    kc = f"{T_SI_R * 1e6:g}"
    out: dict[str, Any] = {"chosen": T_SI_R, "level1": {}, "resolved": {}}
    l1 = r1.get("tsi_level1", {})
    base = l1.get(kc)
    for k, v in l1.items():
        if not isinstance(v, dict) or "L_diff" not in v:
            continue
        row = {"L_diff": v["L_diff"], "Q_diff": v["Q_diff"],
               "n_unknowns": v.get("grid", {}).get("n_unknowns")}
        if base:
            row["dL_rel_to_chosen"] = v["L_diff"] / base["L_diff"] - 1.0
            row["dQ_rel_to_chosen"] = v["Q_diff"] / base["Q_diff"] - 1.0
        if k in ref:
            row["referee_dL_rel_to_chosen"] = ref[k]["deembedded"] / ref[kc]["deembedded"] - 1.0
        out["level1"][k] = row
    lv = r1.get("levels", {}).get(str(RESOLVED))
    alt = r1.get("tsi_resolved", {})
    if isinstance(lv, dict) and "L_diff" in lv and alt.get("L_diff"):
        k = f"{T_SI_SENS_L2 * 1e6:g}"
        out["resolved"] = {
            "level": RESOLVED, "alt_t_si": T_SI_SENS_L2,
            "L_diff": {k: alt["L_diff"], kc: lv["L_diff"]},
            "Q_diff": {k: alt["Q_diff"], kc: lv["Q_diff"]},
            "dL_rel": alt["L_diff"] / lv["L_diff"] - 1.0,
            "dQ_rel": alt["Q_diff"] / lv["Q_diff"] - 1.0,
            "referee_dL_rel": ref[k]["deembedded"] / ref[kc]["deembedded"] - 1.0}
    return out


def sweep_block(r1: dict[str, Any]) -> dict[str, Any]:
    sw = r1.get("sweep", {})
    rows = {k: {kk: v[kk] for kk in ("freq", "L_diff", "Q_diff", "L_se", "Q_se", "L_raw",
                                     "reciprocity_worst", "passivity_worst", "seconds")}
            for k, v in sw.items() if "L_diff" in v}
    if not rows:
        return {}
    ks = sorted(rows, key=float)
    f = [rows[k]["freq"] for k in ks]
    lvals = [rows[k]["L_diff"] for k in ks]
    q = [rows[k]["Q_diff"] for k in ks]
    iq = int(np.argmax(q))
    upper = [i for i, fr in enumerate(f) if fr >= FREQ - 1.0]
    out = {"level": RESOLVED, "rows": rows, "freqs_ghz": [float(k) for k in ks],
           "L_diff": lvals, "Q_diff": q, "q_peak_freq": f[iq], "q_peak": q[iq],
           "L_min_freq": f[int(np.argmin(lvals))],
           "L_rise_2p45_to_top": lvals[-1] / rows[ks[upper[0]]]["L_diff"] - 1.0 if upper else None,
           "srf_fit_upper": srf_fit([f[i] for i in upper], [lvals[i] for i in upper])
           if len(upper) >= 2 else None,
           "srf_fit_what": "1/L = (1 - (f/f0)^2)/L0 fitted to the sweep points f >= 2.45 GHz "
                           "(below 2.45 GHz the Leontovich sheet's internal inductance, which "
                           "falls like 1/sqrt(f), bends L the other way); f0 is the lumped-LC "
                           "self-resonance the rise implies -- a trend, not a resolved "
                           "resonance (the sweep stops at 8 GHz)"}
    return out


def evaluate_gates(study: dict[str, Any]) -> dict[str, Any]:
    g: dict[str, Any] = {}
    solves: dict[str, dict[str, float]] = {}
    for part in ("probe_r0", "levels", "records", "gradients", "fd", "walls_cpu", "remesh_cpu"):
        collect_s(part, study.get(part, {}), solves)
    g["R2"] = r2_gate(solves)
    cpu = ("walls_cpu", "remesh_cpu")
    gpu = [v for k, v in solves.items() if not k.startswith(cpu)]
    g["R2"]["n_solves_cpu"] = len(solves) - len(gpu)
    g["R2"]["reciprocity_worst_gpu"] = max((v["reciprocity"] for v in gpu), default=None)
    g["R2"]["reciprocity_worst_cpu"] = max((v["reciprocity"] for k, v in solves.items()
                                            if k.startswith(cpu)), default=None)
    gr = study.get("gradients", {})
    fd = study.get("fd", {})
    checked: dict[str, Any] = {}
    for name, rec in fd.items():
        if "fd_L" not in rec or not gr:
            continue
        k = rec["k"]
        for q, key in (("L", "dL"), ("Q", "dQ")):
            ad = gr[key][k]
            fdv = rec[f"fd_{q}"]
            checked[f"d{q}_d{name}"] = {"ad": ad, "fd4": fdv, "rel": abs(ad - fdv) / abs(fdv)}
    worst = max((v["rel"] for v in checked.values()), default=None)
    g["R3"] = {"level": RESOLVED, "fd_order": 4, "fd_step_rel": FD_STEP_REL,
               "tolerance": AD_FD_TOL, "checked": checked, "n_checked": len(checked),
               "worst_rel": worst,
               "what": "jax.vjp gradient of the de-embedded L_diff and Q_diff through cuDSS "
                       "against a 4th-order central difference of the SAME discrete model "
                       "(1 % steps: they move L and Q by ~1 %, six decades above the ~1e-9 "
                       "LU noise floor)",
               "passed": worst is not None and len(checked) >= 2 and worst <= AD_FD_TOL}
    return g


def fixture_record() -> dict[str, Any]:
    return {
        "dut": "SQUARE 3-turn spiral (rfx.fdfd.gds.rect_spiral) on TopMetal2 with a TopMetal1 "
               "underpass -- the paper's (r_out, W, S), NOT its symmetric octagon",
        "n_turns": N_TURNS, "theta0": list(THETA0), "params": list(PARAMS), "lead": LEAD,
        "margin": MARGIN,
        "stack": {"t_si": T_SI_R, "si_grading_ratio": SI_GRADE, "t_ox_low": T_OX_LOW,
                  "t_m1": T_M1, "t_via": T_VIA, "t_m2": T_M2, "t_ox_high": T_OX_HIGH,
                  "t_air": T_AIR, "base_dz": BASE_DZ, "metal_cells_level1": METAL_CELLS_R,
                  "eps_si": EPS_SI, "eps_ox": EPS_OX, "sigma_si": SIGMA_SI,
                  "sigma_metal": SIGMA_METAL, "metal_model": "Leontovich surface impedance "
                  "on every metal cell (strip, underpass, via, lead columns, short standard)"},
        "invariant_fixture": {"wall_w": WALL_W, "wall_margin_m": WALL_W * WIDTH,
                              "lid_above_stack_m": WALL_W * WIDTH, "short_gap_m": SHORT_GAP,
                              "port_gap_m": PORT_GAP, "pad_ratio": PAD_RATIO,
                              "refinement": "nested joint (x, y, z) refinement of level 1"},
        "freq": FREQ, "sweep_ghz": list(SWEEP_GHZ), "levels": list(LEVELS),
        "resolved_level": RESOLVED, "finer_level": FINER, "fd_step_rel": FD_STEP_REL,
        "deviations_from_H_kept": ["square, single-ended spiral (not the paper's symmetric "
                                   "octagon)", "TopVia2 widened to 2 um",
                                   "passivation eps lumped into the oxide's 4.1",
                                   "PEC ground under the silicon"],
    }


# ---------------------------------------------------------------------------
# 6. figure

def figure(study: dict[str, Any], path: pathlib.Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 3, figsize=(16, 8.5))
    ax = axs[0, 0]
    lv = study.get("levels", {})
    if lv:
        ks = sorted(lv, key=int)
        cw = [lv[k]["grid"]["cells_across_width"] for k in ks]
        ax.plot(cw, [lv[k]["L_diff"] * 1e9 for k in ks], "o-", color="#1f5fa8", label="L_diff")
        ax.plot(cw, [lv[k]["L_se"] * 1e9 for k in ks], "o:", color="#1f5fa8", label="L_se")
        ax2 = ax.twinx()
        ax2.plot(cw, [lv[k]["Q_diff"] for k in ks], "s-", color="#c2571a", label="Q_diff")
        ax2.plot(cw, [lv[k]["Q_se"] for k in ks], "s:", color="#c2571a", label="Q_se")
        ax2.set_ylabel("Q", color="#c2571a")
        ax.set_xlabel("cells across W (level 1, level 2)")
        ax.set_xticks(cw)
        ax.set_ylabel("L (nH)", color="#1f5fa8")
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="center left")
        ax.set_title("(a) 2.45 GHz by level (paper values)")
    ax = axs[0, 1]
    sw = study.get("sweep", {})
    if sw:
        f = sw["freqs_ghz"]
        ax.plot(f, np.asarray(sw["L_diff"]) * 1e9, "o-", color="#1f5fa8")
        ax.set_xlabel("f (GHz)")
        ax.set_ylabel("L_diff (nH)", color="#1f5fa8")
        ax2 = ax.twinx()
        ax2.plot(f, sw["Q_diff"], "s--", color="#c2571a")
        ax2.set_ylabel("Q_diff", color="#c2571a")
        fit = sw.get("srf_fit_upper") or {}
        ax.set_title(f"(b) sweep, level {RESOLVED}; LC fit f0 = "
                     f"{fit.get('f_self_resonance', float('nan')) / 1e9:.1f} GHz")
    ax = axs[1, 0]
    ts = study.get("tsi_sensitivity", {}).get("level1", {})
    if ts:
        ks = sorted(ts, key=float)
        ax.plot([float(k) for k in ks], [100 * ts[k].get("dL_rel_to_chosen", np.nan) for k in ks],
                "o-", label="FDFD L_diff (level 1)")
        ax.plot([float(k) for k in ks], [100 * ts[k].get("dQ_rel_to_chosen", np.nan) for k in ks],
                "s-", label="FDFD Q_diff (level 1)")
        ax.plot([float(k) for k in ks],
                [100 * ts[k].get("referee_dL_rel_to_chosen", np.nan) for k in ks], "k--",
                label="referee ground image")
        rv = study.get("tsi_sensitivity", {}).get("resolved", {})
        if rv:
            ax.plot([T_SI_SENS_L2 * 1e6], [100 * rv["dL_rel"]], "D", color="#1f5fa8", mfc="none",
                    ms=9, label=f"FDFD L_diff (level {RESOLVED})")
            ax.plot([T_SI_SENS_L2 * 1e6], [100 * rv["dQ_rel"]], "D", color="#ff7f0e", mfc="none",
                    ms=9, label=f"FDFD Q_diff (level {RESOLVED})")
        ax.axvline(T_SI_R * 1e6, color="0.6", lw=0.8)
        ax.set_xlabel("t_si (um)")
        ax.set_ylabel("change vs 190 um (%)")
        ax.legend(fontsize=7)
        ax.set_title("(c) substrate thickness")
    ax = axs[1, 1]
    chk = study.get("gates", {}).get("R3", {}).get("checked", {})
    if chk:
        names = list(chk)
        ax.bar(range(len(names)), [max(chk[n]["rel"], 1e-16) for n in names], color="#3b7d3b")
        ax.axhline(AD_FD_TOL, color="r", ls="--", lw=1, label="1e-4")
        ax.set_yscale("log")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, fontsize=8)
        ax.set_ylabel("|AD - FD4| / |FD4|")
        ax.legend(fontsize=8)
        ax.set_title(f"(d) gradients vs FD4, level {RESOLVED}")
    ax = axs[0, 2]
    wb = study.get("walls", {})
    cases = study.get("walls_cpu", {}).get("cases", {})
    if wb.get("per_point"):
        for p, col in (("nominal", "#1f5fa8"), ("design", "#c2571a"), ("r_out_hi", "#3b7d3b")):
            rows = sorted((c["wall_w"], c["L_diff"]) for c in cases.values() if c["point"] == p)
            ref = dict(rows).get(WALL_W)
            if rows and ref:
                ax.plot([w for w, _ in rows], [100 * (v / ref - 1) for _, v in rows], "o-",
                        color=col, label=f"{p} {tuple(round(t * 1e6, 1) for t in wb['points'][p])} um")
        lp = wb.get("ladder_paper_protocol", {}).get("L", {})
        if lp:
            ks = sorted(lp, key=float)
            ax.plot([float(k) for k in ks], [100 * (lp[k] / lp["10"] - 1) for k in ks], "k--",
                    label="ladder, paper geometry (vacuum, 120 um)")
        ax.axhline(0, color="0.6", lw=0.8)
        ax.set_xlabel("walls and lid (W from the DUT box)")
        ax.set_ylabel("L_diff vs 10 W (%)")
        ax.legend(fontsize=7)
        ax.set_title("(e) wall check, level 1 (CPU)")
    ax = axs[1, 2]
    rm = study.get("remesh", {})
    rows = []
    if rm:
        rows.append(("fresh vs deformed\nmesh (level 1)", rm["dL_rel_fresh_vs_deformed"],
                     rm["dQ_rel_fresh_vs_deformed"]))
        st12 = rm.get("level1_to_2_at_theta_star")
        if st12:
            rows.append(("level 1 -> 2", st12["dL_rel"], st12["dQ_rel"]))
    dp = wb.get("per_point", {}).get("design", {})
    if "dL_rel_10_to_20W" in dp:
        rows.append(("walls 10 -> 20 W\n(level 1)", dp["dL_rel_10_to_20W"], dp["dQ_rel_10_to_20W"]))
    if rows:
        x = np.arange(len(rows))
        ax.bar(x - 0.2, [100 * r[1] for r in rows], 0.4, color="#1f5fa8", label="L_diff")
        ax.bar(x + 0.2, [100 * r[2] for r in rows], 0.4, color="#c2571a", label="Q_diff")
        ax.set_xticks(x)
        ax.set_xticklabels([r[0] for r in rows], fontsize=8)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_ylabel("relative change (%)")
        ax.legend(fontsize=8)
        ax.set_title("(f) at the design point theta*")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="study R parts A and B (assembly)")
    ap.add_argument("--from-json", action="store_true",
                    help="reuse the referee block of the JSON on disk")
    ap.add_argument("--no-figure", action="store_true")
    ap.add_argument("--walls", type=int, nargs="?", const=-1, default=None, metavar="N",
                    help="solve (at most N of) the missing CPU wall-check cases into the JSON "
                         "and exit (all of them without N)")
    ap.add_argument("--remesh", action="store_true",
                    help="solve the fresh-mesh case at theta* into the JSON and exit")
    args = ap.parse_args(argv)
    import jax
    jax.config.update("jax_enable_x64", True)
    if args.walls is not None:
        n = run_wall_check(None if args.walls < 0 else args.walls)
        print(f"solved {n} wall-check case(s)")
        return 0
    if args.remesh:
        print(f"solved {run_remesh_check()} remesh case(s)")
        return 0
    reuse = json.loads(JSON_PATH.read_text()) if args.from_json and JSON_PATH.exists() else None
    study = assemble(reuse)
    JSON_PATH.write_text(json.dumps(study, indent=1, default=float) + "\n")
    print(f"wrote {JSON_PATH}")
    for k, v in study["gates"].items():
        print(f"  {k}: passed={v.get('passed')}")
    if not args.no_figure:
        figure(study, PNG_PATH)
        print(f"wrote {PNG_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
