"""Validation study D2b: a STRAIGHT BAR on the spiral fixture, to decompose
the residual gap of ``validation/fdfd/spiral_convergence.py``.

Why this study exists
---------------------
The completed spiral study removed every model difference it could find
(volumetric uniform-current metal, graded wall padding, area-exact corner
convention, matched reference plane) and still sits 10-12 % BELOW the
Greenhouse referee after Richardson extrapolation, with an observed order
of 1.5 and a lead-length differential ratio of 0.76 at ``base_dx = W``.
Two families of explanation survive:

(i)  the Yee grid's representation of the inductance of a STRIP resolved by
     1-3 cells across its cross-section (a discretisation error distributed
     along the conductor), plus whatever the open/short de-embedding leaves
     behind at the reference plane;
(ii) something spiral-specific: corners, the underpass, the via, the
     turn-to-turn mutual terms.

A straight bar between the two lead columns has none of (ii) -- no corner,
no underpass, no via, no mutual coupling of parallel runs -- while using
literally the same build, the same three fixtures, the same open/short
de-embedding and the same traced body-fitted metric (``dut_kind="bar"``, an
additive hook in :mod:`rfx.fdfd.spiral`). Its referee is a ONE-LINE
Greenhouse sum: the self partial inductance of a rectangular bar plus its
PEC-ground image, with no corner convention to argue about (the bar's plan
area is ``l x W`` exactly).

The DUT (``rfx.fdfd.spiral``, ``dut_kind="bar"``)
-------------------------------------------------
One strip of width ``W`` and thickness 2 um on the top metal, spanning
``x in [-(l+W)/2, (l+W)/2]``, ``y in [-W/2, W/2]``; the two lead columns
drop from its two end squares (``|x| in [(l-W)/2, (l+W)/2]``) to one cell
above the PEC ground, where the lumped ports sit. ``theta = (l_bar, W)``
-- a TWO-tuple, not the spiral's ``(r_out, spacing, width)`` with an
ignored entry -- and both entries are breakpoints of the traced metric, so
``jax.grad`` gives ``dL/dl_bar`` and ``dL/dW`` of the discrete model.
OPEN = the two columns; SHORT = the columns, the bar row on M2 and the PEC
post to ground between them, exactly as ``build_spiral`` builds them for
the spiral.

The reference plane (the one thing that is NOT the same as the spiral)
----------------------------------------------------------------------
``build_spiral``'s short standard grounds each lead through a post that
starts ONE CELL inside the lead column (``post_i0 = inner.i1 + 1``,
``post_i1 = outer.i0 - 1``; the isolating cell is mandatory -- a post
touching the column would make the port's own Ez edges PEC). For the
SPIRAL that post is perpendicular to the lead, so it shifts the reference
plane by nothing and the study put the plane at the lead-column footprint
CENTRE, with a +-W/2 sensitivity of +-1.46 %. For the BAR the post is
COLLINEAR with the DUT current, so the short standard removes, at each end,
the strip from the footprint centre to the post edge: ``W/2 + base_dx``.
Three plane conventions are therefore reported at every level:

  ``centres``     ``l_eff = l``            the spiral study's convention
  ``inner_edges`` ``l_eff = l - W``        planes at the footprint's DUT-side edge
  ``post_edges``  ``l_eff = l - W - 2 dx`` where the short standard actually grounds

``centres`` is the PRIMARY (the gates quote it) because it is the spiral
study's convention and because it is grid-independent; ``inner_edges`` is
its grid-independent h -> 0 limit and ``post_edges`` the geometric one at a
finite grid. The band ``l +- W`` (one footprint half-width per plane) is
this fixture's analogue of the spiral's +-1.46 % and is reported as
``reference_plane_sensitivity``; it is LARGE here -- -11.39 % / +11.43 % at
``l = 100 um``, ``W = 10 um`` and -23.22 % / +23.45 % at ``l = 100 um``,
``W = 20 um`` -- because a bar 5 to 20 widths long is short. THIS IS WHY
THE DIFFERENTIAL (below) IS THE PRIMARY MEASUREMENT and why the absolute
gap B1/B3 is reported against three conventions rather than one.

The differential (the plane-free, fixture-free measurement)
-----------------------------------------------------------
``[L(l2) - L(l1)]_FDFD / [L(l2) - L(l1)]_referee`` at ``l2 = 2 l1``. Every
fixture residual -- the ports, the columns, the de-embedding, the reference
plane offset, to first order the walls -- is common to the two solves and
cancels, so the ratio is the FDFD's inductance per unit length of strip
against the exact uniform-current value. It is the direct analogue of the
spiral study's lead-length differential (ratio 0.763 at W/1) and it is what
decides between explanation (i) and (ii).

The walls are NOT level-independent at ``pad_cells = 4`` (a finding)
--------------------------------------------------------------------
``spiral.pad_lines`` grows its ``pad_cells`` padding cells geometrically
from the LAST CELL OF THE MESHED BOX, whose size is ``base_dx``. With
``pad_cells = 4`` and ``pad_ratio = 1.5`` the padding therefore reaches
``12.19 base_dx`` beyond the box -- i.e. the PEC walls CREEP IN in
proportion to the refinement. Measured here on the ``l = 100 um``,
``W = 10 um`` bar: at W/1 the y wall is 137 um out and ``pad_cells`` 4 -> 8
-> 12 gives 27.465 -> 27.707 -> 27.709 pH (+0.88 %, then converged), while
at W/3 the y wall is only 56 um out and 4 -> 8 gives 38.026 -> 40.000 pH
(+5.19 %). So a ``pad_cells``-fixed refinement sequence carries a wall bias
that GROWS with refinement and partly cancels the discretisation error it
is trying to measure. The cheap escape -- fewer, faster-growing padding
cells -- does not work: ``pad_cells = 4`` with ``pad_ratio`` raised to
2.67 reaches the same 865 um wall but gives 26.38 pH at W/1 against the
27.71 pH of the gentle 8-cell padding at 804 um, i.e. the aggressive
grading contributes an error of its own, five times the wall error it
removes. Therefore this study keeps ``pad_cells = 4``, ``pad_ratio = 1.5``
(identical to the spiral study) for the primary series AND measures a wall
ladder (``pad_cells`` 4 / 6 / 8) at every level of the cheapest width, from
which a wall-extrapolated value and a wall-corrected differential ratio are
reported.

Protocol (identical to the spiral study unless stated)
------------------------------------------------------
100 MHz, ``sigma_volumetric = 3e6`` S/m (skin depth 29.1 um: 14.6 x the
2 um metal thickness and >= 1.5 x the widths used, so the current density is
uniform and L carries the DC internal inductance, as the referee does),
VACUUM dielectrics (``eps_si = eps_ox = 1``: the referee is
magnetoquasistatic), the same small stack (20 um Si + 1 um oxide + M1 + via
+ M2, ground to M2 centre = 26 um), ``base_dz = 10 um``,
``metal_cells = 2``, ``margin = 10 um``, ``port_gap_cells = 1``,
``pad_cells = 4``, ``pad_ratio = 1.5``. Levels ``base_dx = W/1, W/2, W/3,
W/4, W/6`` skipped where N would exceed 112k. Two lengths (100 and 200 um)
and two widths (10 and 20 um) at the fixed 2 um thickness.
At ``W = 20 um`` the W/1 and W/2 grids are IDENTICAL (the 10 um margin is a
mandatory line and ``grade_lines``' 1.5 neighbour-ratio rule already splits
the 20 um cells next to it), so W/2 is dropped there.

Run::

    .venv/bin/python validation/fdfd/straight_bar_convergence.py
        [--widths 10 20] [--lengths 100 200] [--levels 1 2 3 4 6]
        [--n-max 112000] [--no-walls] [--no-systematics] [--from-json]

writes ``straight_bar_convergence.json`` and ``.png`` next to this file;
the JSON is rewritten after every single solve, so an interrupted run keeps
everything it had measured.

Gates and what they measured (every number below is in the JSON)
-----------------------------------------------------------------
B1  per-level gap of ``L_dut`` against the referee at all three plane
    conventions, plus the observed Richardson order. At ``pad_cells = 4``
    the gap shrinks monotonically only to W/3 and then STALLS or reverses
    (W = 10 um, l = 100 um: -46.1 / -29.5 / -25.3 / -24.4 / -25.4 % at
    W/1..W/6 against the ``centres`` plane), because the walls creep in.
    With the wall ladder extrapolated away it keeps shrinking:
    -45.6 / -27.6 / -21.2 % against ``centres`` and
    -17.6 / -6.3 / -2.7 % against ``post_edges`` -- the plane the short
    standard actually grounds. Same picture at W = 20 um, where the
    wall-corrected ``post_edges`` gap is -1.0 / -1.1 / +1.1 % at 3 / 4 / 6
    cells across the width (and -1.0 / -1.5 % for l = 200 um). So the FDFD
    plus the open/short de-embedding reproduce the EXACT uniform-current
    inductance of a strip over a PEC ground -- image term included -- to
    1-3 % from three cells across the width, once the walls are removed and
    the plane is taken where the short puts it.
B2  the differential ratio. RAW (``pad_cells = 4``, the spiral study's
    protocol) it FAILS and is non-monotone: 0.814 / 0.893 / 0.889 / 0.872
    at W = 10 um (W/1..W/4) and 0.908 / 0.936 / 0.930 / 0.912 at W = 20 um
    (W/1, W/3, W/4, W/6). WALL-CORRECTED it converges:
    0.829 / 0.932 / 0.975 at W = 10 um (1 / 2 / 3 cells across the width)
    and 0.925 / 0.981 / 0.975 / 1.004 at W = 20 um (1 / 3 / 4 / 6 cells),
    i.e. it PASSES the 3 % gate from three cells on. Its own plane
    sensitivity is small, as the measurement requires: taking the referee
    difference at the ``post_edges`` plane instead of ``centres`` moves the
    ratio by at most 2.6 % (0.908 -> 0.933) and usually under 1 %.
B3  the Richardson range (p = 1, p = 2, observed order). Against the
    ``centres`` plane it FAILS by a wide margin at ``pad_cells = 4`` (the
    plane, not the field solution). The wall-extrapolated series over the
    levels with a full three-point ladder gives, for W = 10 um /
    l = 100 um, an observed order of 0.89 and a range [42.73, 47.61] pH
    that BRACKETS the referee at the ``inner_edges`` plane (45.12 pH,
    -5.3 % / +5.5 %), and for W = 20 um / l = 100 um [29.29, 31.79] pH
    against 30.36 pH (-3.5 % / +4.7 %). The per-level ``post_edges``
    statement of B1 is the tighter and better-posed one.
B4  ``jax.grad`` vs FD4 (1 % steps) for ``dL/dl_bar`` and ``dL/dW``:
    worst 1.49e-6 at W = 20 um, l = 100 um, W/3 (dL/dl_bar = +3.7212e-7,
    dL/dW = -7.5679e-7 H/m), and 1.13e-6 / 2.04e-6 at W/1. PASSES (1e-4).

Protocol checks
---------------
*  sigma plateau: ``L_dut`` = 23.348 / 23.354 / 23.395 / 23.769 / 25.058 /
   27.492 pH at sigma = 3e5 / 1e6 / 3e6 / 1e7 / 3e7 / 1e8 S/m (W = 20 um,
   l = 100 um, W/3), so ``SIGMA_VOL = 3e6`` is on the uniform-current
   plateau (+0.18 % from 1e6) and 1e7 is already +1.6 % off it. PEC gives
   27.745 pH, +18.6 % ABOVE the plateau -- the OPPOSITE sign to the
   spiral's -6.7 %, and reproduced from the other end by the volumetric
   model itself (27.49 pH at sigma = 1e8, skin depth 5.0 um).
*  the wall extrapolation is itself validated: the 3-point Richardson in
   ``1 / wall_distance`` from ``pad_cells`` 4 / 6 / 8 gives 27.7168 pH at
   W = 10 um, l = 100 um, W/1 against a DIRECT ``pad_cells = 12`` solve
   (walls 3872 um out, N = 73960) of 27.7095 pH: +0.026 %. Only levels with
   a full three-point ladder feed the wall-corrected series; a two-point
   one overshoots (pad 4/6 alone gives +10.4 % at W/4 against +5.5 % for
   4/6/8 one level coarser) and is excluded.
*  ``metal_cells`` 2 -> 4 (2 -> 4 cells across the 2 um thickness): +0.13 %.
   ``base_dz`` 10 -> 5 -> 2.5 um (the grid between the strip and the ground,
   held fixed by the protocol): +1.60 % / +0.28 %. Frequency flatness
   50-200 MHz: 2e-7.

Conclusion: where the spiral's 10-12 % goes
--------------------------------------------
1. The deficit is NOT spiral-specific. A bar with no corner, no underpass,
   no via and no turn-to-turn mutual shows the same slow, distributed,
   per-unit-length deficit at the same resolutions and under the same
   protocol. Measured side by side here (same ``pad_cells = 4``, same
   stack, same sigma, same de-embedding): the bar's differential ratio is
   0.814 (W/1) and 0.893 (W/2) at W = 10 um, the SPIRAL's own lead-length
   differential -- remeasured in this session, not quoted -- is 0.770 (W/1)
   and 0.883 (W/2). Two geometries that share nothing but the strip
   cross-section and the fixture agree to 4 points at W/1 and 1 point at
   W/2.
2. It is the strip CROSS-SECTION discretisation, and it converges. With the
   PEC walls held at a fixed distance the bar's inductance per unit length
   is 17 % low at ONE cell across the width, 7 % low at two, 2.5 % low at
   three and 0.4 % HIGH at six (2 cells across the 2 um thickness
   throughout). So ``base_dx = W/3`` with ``metal_cells = 2`` is what a 3 %
   inductance needs, and W/6 buys 1 %. The thickness is not the limiter
   (``metal_cells`` 2 -> 4 is +0.13 %); the vertical grid between the strip
   and the ground is worth more (+1.6 % for ``base_dz`` 10 -> 5 um) and is
   held fixed by this protocol, so it is a floor on both studies.
3. The walls are the BAR's problem, not the spiral's. ``pad_cells = 4``
   padding reaches only ``12.19 base_dx``, so an open bar's wall bias grows
   from +0.9 % (W/1) to +5.8 % (W/6) and masks the convergence entirely.
   The same ladder run on the SPIRAL fixture in this session gives +0.12 %
   (W/1) and +0.39 % (W/2) -- a closed loop is a magnetic dipole and its
   field dies as 1/r^3 -- so the spiral study's Richardson is NOT
   wall-contaminated and this finding does not move its numbers. It does
   mean the ``pad_cells`` protocol must not be reused for open structures
   without re-gating the walls at every level.
4. What the bar adds about the reference plane. For the spiral the short
   standard's bar leaves the lead column PERPENDICULAR to the DUT current,
   so the study put the plane at the footprint centre with a +-1.46 % band.
   For the bar the same bar is COLLINEAR and the measurement is
   unambiguous: the de-embedded conductor is the one between the short's
   ground POSTS, ``l - W - 2 base_dx``, and only at that plane does the
   extracted L match the referee (B1 above). Applied to the spiral, the
   plane sits ``base_dx`` further in than assumed at each of its two
   terminals, worth ``2 base_dx`` of conductor = 2.9 % of L at W/1 and
   1.0 % at W/3 (from the measured referee 0.501 pH/um) -- larger than the
   +-1.46 % band the spiral study quoted, and in the direction that closes
   the gap.
   Adding it up at the spiral's W/2: 11.7 % from the strip (its own
   differential), ~1.4 % from the plane and 1.2 % from the corner
   convention account for ~14.3 % of its measured -16.4 %. The unexplained
   remainder is a few points and, from these numbers, the only structures
   left to carry it are the NINE right-angle corners of that 10-bar chain
   (9 W^2 = 9.0e-10 m^2 of the 5.97e-9 m^2 strip polygon, i.e. 15 % of the
   metal, and the one place where a staircase Yee grid misplaces both the
   path length and the local self term) and the via, which the referee does
   not model at all. This study cannot separate those two; a corner-only
   DUT on the same fixture would, and is the obvious next decomposition.

Scope fence. One frequency, one thickness, one stack, one ground height;
in-plane refinement only (the vertical grid is held fixed and its
sensitivity is reported, not extrapolated); the referee is
magnetoquasistatic with uniform current density and an infinite PEC ground.
The wall-corrected series carries the +0.026 % (3-point, validated) or
worse (2-point, excluded) uncertainty of its extrapolation. The spiral
blocks rebuild that fixture from its published parameters -- they never
import the spiral study -- and reproduce its W/1 ``pad_cells = 4``
``L_diff`` of 262.439 pH exactly. Sizes are CPU-SuperLU sizes and the
timings were taken with two other solver jobs on the same machine, so they
are upper bounds.

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
JSON_PATH = HERE / "straight_bar_convergence.json"
PNG_PATH = HERE / "straight_bar_convergence.png"

# ---- the fixture (both tools), copied from the spiral study ----------------
FREQ = 1e8
MARGIN = 10e-6
T_SI, T_OX_LOW, T_M1, T_VIA, T_M2, T_OX_HIGH, T_AIR = 20e-6, 1e-6, 2e-6, 2e-6, 2e-6, 3e-6, 10e-6
BASE_DZ = 10e-6            # vertical grid, the SAME at every level
METAL_CELLS = 2            # >= 2 cells across the metal thickness
PAD_CELLS, PAD_RATIO = 4, 1.5
SIGMA_VOL = 3e6            # uniform-current plateau (measured in the spiral study)
GROUND_H = T_SI + T_OX_LOW + T_M1 + T_VIA + 0.5 * T_M2          # 26 um, ground -> M2 centre
FD_STEP_REL = 0.01         # 1 % steps (B4 measures the agreement they give)
N_MAX = 112_000            # single-solve guideline on this machine (~6 GB RSS)

WIDTHS = (10e-6, 20e-6)
LENGTHS = (100e-6, 200e-6)
LEVELS = (1.0, 2.0, 3.0, 4.0, 6.0)
WALL_PADS = (4, 6, 8)
WALL_WIDTH = 20e-6         # the wall ladder runs on the cheapest width
PLANES = ("centres", "inner_edges", "post_edges")


def case_key(w: float, ell: float) -> str:
    return f"W{w * 1e6:g}_l{ell * 1e6:g}"


# ----------------------------------------------------------------------------
# 1. referee (host numpy, loaded from its file; never imported by rfx)

def load_referee():
    spec = importlib.util.spec_from_file_location("spiral_greenhouse", REFEREE_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def referee_bar(sg, l_eff: float, width: float, thickness: float = T_M2,
                ground: float | None = GROUND_H) -> Any:
    """Greenhouse value of ONE straight bar of length ``l_eff``, width
    ``width``, thickness ``thickness``, centred on the origin with its
    centreline at z = 0 and (by default) a PEC ground ``GROUND_H`` below:
    ``self_sum`` is the Hoer-Love self partial inductance, ``mutual_sum`` is
    identically zero (one segment) and ``image_sum`` the negative coupling
    to the mirror bar. No corner convention exists, so the area gate is
    trivial: the bar's plan area is ``l_eff * width`` exactly."""
    seg = sg.Segment((-0.5 * l_eff, 0.0, 0.0), (0.5 * l_eff, 0.0, 0.0), width, thickness)
    return sg.greenhouse_terms([seg], ground_height=ground)


def plane_lengths(ell: float, width: float, base_dx: float) -> dict[str, float]:
    """The three reference-plane conventions (see the module doc)."""
    return {"centres": ell, "inner_edges": ell - width, "post_edges": ell - width - 2.0 * base_dx}


def referee_block(sg, ell: float, width: float) -> dict[str, Any]:
    g = referee_bar(sg, ell, width)
    free = referee_bar(sg, ell, width, ground=None)
    band = {f"{s:+g}W": referee_bar(sg, ell + s * width, width).total for s in (-1.0, 0.0, 1.0)}
    return {
        "bar_length": ell, "width": width, "thickness": T_M2, "ground_height": GROUND_H,
        "L_centres": g.total, "self_sum": g.self_sum, "mutual_sum": g.mutual_sum,
        "image_sum": g.image_sum, "worst_error": g.worst_error,
        "L_free_space": free.total, "ground_shielding": g.total / free.total - 1.0,
        "L_inner_edges": referee_bar(sg, ell - width, width).total,
        "plane_band": band,
        "reference_plane_sensitivity": [band["-1W"] / g.total - 1.0, band["+1W"] / g.total - 1.0],
        "plan_area": ell * width,
    }


def invert_referee(sg, value: float, width: float, lo: float = 1e-6, hi: float = 2e-3) -> float:
    """The bar length whose referee inductance is ``value`` (bisection on a
    monotone function): the "implied de-embedded length" of an FDFD result.
    Its offset from ``l`` says whether the FDFD gap is a constant reference
    plane shift (same offset for both lengths) or distributed along the
    conductor (offset growing with length)."""
    f = lambda x: referee_bar(sg, x, width).total - value          # noqa: E731
    a, b = lo, hi
    if f(a) > 0 or f(b) < 0:
        return float("nan")
    for _ in range(200):
        m = 0.5 * (a + b)
        if f(m) < 0:
            a = m
        else:
            b = m
    return 0.5 * (a + b)


# ----------------------------------------------------------------------------
# 2. the FDFD side

def stack(base_dz: float = BASE_DZ, metal_cells: int = METAL_CELLS, t_air: float = T_AIR):
    from rfx.fdfd import spiral as sm
    return sm.SmallStack(t_si=T_SI, t_ox_low=T_OX_LOW, t_m1=T_M1, t_via=T_VIA, t_m2=T_M2,
                         t_ox_high=T_OX_HIGH, t_air=t_air, base_dz=base_dz,
                         metal_cells=metal_cells, eps_si=1.0, eps_ox=1.0)


def build_bar(ell: float, width: float, div: float, pad: int = PAD_CELLS,
              base_dz: float = BASE_DZ, metal_cells: int = METAL_CELLS,
              margin: float = MARGIN, t_air: float = T_AIR, pad_ratio: float = PAD_RATIO):
    """The static straight-bar model at ``base_dx = width / div``."""
    from rfx.fdfd import spiral as sm
    spec = sm.SpiralSpec(width=width, base_dx=width / div, margin=margin,
                         stack=stack(base_dz, metal_cells, t_air), port_gap_cells=1,
                         pad_cells=pad, pad_ratio=pad_ratio,
                         dut_kind="bar", bar_length=ell)
    return sm.build_spiral(spec)


def l_dut(model, theta=None, sigma: float | None = SIGMA_VOL):
    """De-embedded ``L_diff`` (H) of the volumetric-metal bar fixture --
    the quantity the referee is compared against. ``sigma=None`` gives the
    PEC fixture."""
    from rfx.fdfd import spiral as sm
    return sm.solve_spiral(model, FREQ, theta=theta, sigma_volumetric=sigma).L_diff


def wall_distance(model, ell: float, width: float) -> float:
    """Smallest distance from the conductor to a PEC wall (the four side
    walls and the lid; the ground is physical)."""
    return float(min(model.x_nom[-1] - 0.5 * (ell + width), -model.x_nom[0] - 0.5 * (ell + width),
                     model.y_nom[-1] - 0.5 * width, -model.y_nom[0] - 0.5 * width,
                     model.z[-1] - (GROUND_H + 0.5 * T_M2)))


def model_record(model, ell: float, width: float) -> dict[str, Any]:
    return {"shape": list(model.shape), "n_unknowns": int(model.n_unknowns),
            "wall_x": [float(model.x_nom[0]), float(model.x_nom[-1])],
            "wall_y": [float(model.y_nom[0]), float(model.y_nom[-1])],
            "lid_z": float(model.z[-1]), "nz": int(model.shape[2]),
            "wall_distance": wall_distance(model, ell, width),
            "build_seconds": float(model.build_seconds)}


def run_case(sg, ell: float, width: float, div: float, pad: int = PAD_CELLS,
             n_max: int = N_MAX, with_pec: bool = True, verbose: bool = True,
             **kw) -> dict[str, Any] | None:
    """One (length, width, level) point: the volumetric-metal de-embedded
    ``L_diff``, the PEC value for contrast, and the gap against all three
    reference-plane conventions. Returns ``None`` when the grid would exceed
    ``n_max`` unknowns (reported as a skip, not a failure)."""
    import jax

    base_dx = width / div
    model = build_bar(ell, width, div, pad=pad, **kw)
    if model.n_unknowns > n_max:
        if verbose:
            print(f"    W/{div:g} pad={pad}: N={model.n_unknowns} > {n_max}, SKIPPED", flush=True)
        return None
    rec: dict[str, Any] = {"div": div, "base_dx": base_dx, "pad_cells": pad,
                           "bar_length": ell, "width": width,
                           "grid": model_record(model, ell, width)}
    t0 = time.time()
    val = l_dut(model)
    jax.block_until_ready(val)
    rec["L_dut"] = float(np.real(val))
    rec["seconds"] = time.time() - t0
    if with_pec:
        t0 = time.time()
        rec["L_pec"] = float(np.real(l_dut(model, sigma=None)))
        rec["pec_seconds"] = time.time() - t0
        rec["L_pec_over_L_uniform_minus_1"] = rec["L_pec"] / rec["L_dut"] - 1.0
    planes = plane_lengths(ell, width, base_dx)
    rec["plane_lengths"] = planes
    rec["L_referee"] = {k: referee_bar(sg, v, width).total for k, v in planes.items()}
    rec["gap"] = {k: rec["L_dut"] / v - 1.0 for k, v in rec["L_referee"].items()}
    rec["implied_length"] = invert_referee(sg, rec["L_dut"], width)
    rec["implied_plane_shift"] = 0.5 * (ell - rec["implied_length"])
    if verbose:
        print(f"    W/{div:g} pad={pad}: N={model.n_unknowns} {tuple(model.shape)} "
              f"wall={rec['grid']['wall_distance'] * 1e6:.0f} um  L={rec['L_dut'] * 1e12:.3f} pH  "
              f"gap(centres)={100 * rec['gap']['centres']:+.2f} % "
              f"gap(post)={100 * rec['gap']['post_edges']:+.2f} %  "
              f"l_implied={rec['implied_length'] * 1e6:.1f} um  ({rec['seconds']:.0f} s)", flush=True)
    return rec


# ----------------------------------------------------------------------------
# 3. numerics (the SAME formulas the spiral study uses, copied so that this
#    file does not depend on another study module)

def fd4(f: Callable[[float], float], x0: float, d: float) -> float:
    return (-f(x0 + 2 * d) + 8 * f(x0 + d) - 8 * f(x0 - d) + f(x0 - 2 * d)) / (12 * d)


def richardson(h: Sequence[float], values: Sequence[float]) -> dict[str, Any]:
    """Extrapolations of ``L(h) = L_inf - C h^p`` to ``h -> 0``: ``p = 1``
    and ``p = 2`` from the finest pair, and the OBSERVED order from the
    finest three levels. Returns every estimate and their range."""
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


# ----------------------------------------------------------------------------
# 4. the differential (the plane-free measurement)

def differential(sg, levels: dict[str, Any], width: float, l1: float, l2: float,
                 wall: dict[str, Any] | None = None) -> dict[str, Any]:
    """``[L(l2) - L(l1)]_FDFD / [L(l2) - L(l1)]_referee`` per level.

    Both solves share the grid resolution, the fixture, the de-embedding and
    the reference-plane offset, so all of those cancel in the difference and
    the ratio is the FDFD's inductance per unit length of strip against the
    referee's. The referee difference is quoted for the ``centres`` and
    ``post_edges`` conventions: shifting BOTH planes by the same delta moves
    the difference only at second order, which is the point of the
    measurement (measured spread below, ``ratio_post_edges``).

    ``wall`` (the wall ladder of this width, if measured) supplies a
    per-level multiplicative wall correction ``L_wall_extrapolated /
    L(pad=4)`` for each length; the corrected ratio uses the corrected
    increments."""
    k1, k2 = case_key(width, l1), case_key(width, l2)
    out: dict[str, Any] = {"width": width, "l1": l1, "l2": l2, "levels": {}}
    for key in sorted(set(levels[k1]) & set(levels[k2]), key=float):
        a, b = levels[k1][key], levels[k2][key]
        dx = a["base_dx"]
        d_f = b["L_dut"] - a["L_dut"]
        row: dict[str, Any] = {"base_dx": dx, "div": a["div"],
                               "L_l1": a["L_dut"], "L_l2": b["L_dut"], "dL_fdfd": d_f,
                               "n_unknowns": [a["grid"]["n_unknowns"], b["grid"]["n_unknowns"]]}
        for conv in ("centres", "post_edges"):
            d_r = b["L_referee"][conv] - a["L_referee"][conv]
            row[f"dL_referee_{conv}"] = d_r
            row[f"ratio_{conv}"] = d_f / d_r
        if wall:
            corr = []
            for case_k in (k1, k2):
                we = wall.get(case_k, {}).get(key)
                corr.append(we["L_wall_extrapolated"] / we["L_pad4"]
                            if we and "L_wall_extrapolated" in we else None)
            if all(c is not None for c in corr):
                d_fc = corr[1] * b["L_dut"] - corr[0] * a["L_dut"]
                row["wall_correction"] = corr
                row["dL_fdfd_wall_corrected"] = d_fc
                row["ratio_centres_wall_corrected"] = d_fc / row["dL_referee_centres"]
        out["levels"][key] = row
        print(f"    W/{a['div']:g}: dL_fdfd={d_f * 1e12:+.3f} "
              f"dL_ref={row['dL_referee_centres'] * 1e12:+.3f} pH  "
              f"ratio={row['ratio_centres']:.4f} (post-edge plane {row['ratio_post_edges']:.4f}"
              + (f", wall-corrected {row['ratio_centres_wall_corrected']:.4f}"
                 if "ratio_centres_wall_corrected" in row else "") + ")", flush=True)
    return out


# ----------------------------------------------------------------------------
# 5. the wall ladder (pad_cells 4 / 6 / 8 at every level)

def wall_ladder(sg, ell: float, width: float, divs: Sequence[float], pads: Sequence[int] = WALL_PADS,
                n_max: int = N_MAX, known: dict[str, Any] | None = None) -> dict[str, Any]:
    """``L_dut`` against the number of graded padding cells at every level,
    plus a Richardson extrapolation in ``1 / wall_distance`` (the wall error
    of a magnetic source near a PEC plane decays as a power of the
    distance). The quantity that matters is ``rel_pad4``, how much the
    ``pad_cells = 4`` series used everywhere else is biased at that level."""
    out: dict[str, Any] = {}
    for div in divs:
        rows = []
        for pad in pads:
            r = (known or {}).get(f"{div:g}") if pad == PAD_CELLS else None
            if r is None:
                r = run_case(sg, ell, width, div, pad=pad, n_max=n_max, with_pec=False,
                             verbose=False)
            if r is None:
                continue
            rows.append({"pad_cells": pad, "L_dut": r["L_dut"],
                         "wall_distance": r["grid"]["wall_distance"],
                         "n_unknowns": r["grid"]["n_unknowns"], "seconds": r["seconds"]})
            print(f"    W/{div:g} pad={pad}: N={rows[-1]['n_unknowns']} "
                  f"wall={rows[-1]['wall_distance'] * 1e6:.0f} um  L={rows[-1]['L_dut'] * 1e12:.4f} pH "
                  f"({rows[-1]['seconds']:.0f} s)", flush=True)
        if len(rows) < 2:
            continue
        entry: dict[str, Any] = {"rows": rows, "L_pad4": rows[0]["L_dut"]}
        if len(rows) >= 2:
            rich = richardson([1.0 / r["wall_distance"] for r in rows], [r["L_dut"] for r in rows])
            entry["wall_richardson"] = rich
            entry["L_wall_extrapolated"] = float(np.mean(rich["range"]))
            entry["rel_pad4"] = entry["L_wall_extrapolated"] / entry["L_pad4"] - 1.0
            print(f"      -> wall-extrapolated {entry['L_wall_extrapolated'] * 1e12:.4f} pH "
                  f"({100 * entry['rel_pad4']:+.2f} % on pad=4)", flush=True)
        out[f"{div:g}"] = entry
    return out


# ----------------------------------------------------------------------------
# 6. systematics and the gradient check

def sigma_plateau(ell: float, width: float, div: float,
                  sigmas: Sequence[float] = (3e5, 1e6, 3e6, 1e7, 3e7, 1e8)) -> dict[str, Any]:
    """Protocol gate A, the bar's version: the uniform-current plateau of
    ``L_dut`` in the metal conductivity, with the skin depth of each point
    and the PEC value for contrast. ``SIGMA_VOL = 3e6`` must sit ON the
    low-sigma plateau (where the skin depth swamps the cross-section and the
    current density is uniform, exactly the referee's model); the high-sigma
    end is the skin-confined limit and is reported, not used."""
    from rfx.fdfd import spiral as sm
    m = build_bar(ell, width, div)
    out: dict[str, Any] = {"bar_length": ell, "width": width, "div": div,
                           "n_unknowns": int(m.n_unknowns), "sigma": list(map(float, sigmas)),
                           "L_dut": [], "Q_diff": [], "skin_depth": [],
                           "skin_over_width": [], "skin_over_thickness": []}
    for sig in sigmas:
        r = sm.solve_spiral(m, FREQ, sigma_volumetric=sig)
        d = float(sm.skin_depth(FREQ, sig))
        out["L_dut"].append(float(np.real(r.L_diff)))
        out["Q_diff"].append(float(r.Q_diff))
        out["skin_depth"].append(d)
        out["skin_over_width"].append(d / width)
        out["skin_over_thickness"].append(d / T_M2)
        print(f"    sigma={sig:.0e} delta={d * 1e6:6.1f} um ({d / width:.2f} W): "
              f"L={out['L_dut'][-1] * 1e12:8.4f} pH  Q={out['Q_diff'][-1]:.4f}", flush=True)
    out["L_pec"] = float(np.real(l_dut(m, sigma=None)))
    i = out["sigma"].index(SIGMA_VOL)
    j = out["sigma"].index(1e6)
    out["L_at_sigma_vol"] = out["L_dut"][i]
    out["plateau_rel_change_1e6_to_3e6"] = out["L_dut"][i] / out["L_dut"][j] - 1.0
    out["L_pec_over_plateau_minus_1"] = out["L_pec"] / out["L_dut"][i] - 1.0
    print(f"    PEC: L={out['L_pec'] * 1e12:8.4f} pH "
          f"({100 * out['L_pec_over_plateau_minus_1']:+.2f} % on the plateau); plateau flatness "
          f"1e6 -> 3e6 = {100 * out['plateau_rel_change_1e6_to_3e6']:+.3f} %", flush=True)
    return out


def systematics(sg, ell: float, width: float, div: float) -> dict[str, Any]:
    """Everything on the FDFD side that could fake a model difference, at
    one level: the metal cross-section cells, the vertical grid, the
    frequency flatness and the PEC-vs-uniform-current gap."""
    from rfx.fdfd import spiral as sm
    out: dict[str, Any] = {"bar_length": ell, "width": width, "div": div}
    base_model = build_bar(ell, width, div)
    base = float(np.real(l_dut(base_model)))
    out["L_dut"] = base
    out["n_unknowns"] = int(base_model.n_unknowns)

    out["freq_flatness"] = {}
    for f in (0.5 * FREQ, FREQ, 2.0 * FREQ):
        out["freq_flatness"][f"{f:.3e}"] = float(
            np.real(sm.solve_spiral(base_model, f, sigma_volumetric=SIGMA_VOL).L_diff))
    vals = list(out["freq_flatness"].values())
    out["freq_flatness_rel_spread"] = (max(vals) - min(vals)) / base

    out["metal_cells"] = []
    for mc in (METAL_CELLS, 4):
        m = build_bar(ell, width, div, metal_cells=mc)
        v = float(np.real(l_dut(m)))
        out["metal_cells"].append({"metal_cells": mc, "nz": int(m.shape[2]),
                                   "n_unknowns": int(m.n_unknowns), "L_dut": v,
                                   "rel": v / base - 1.0})
        print(f"    metal_cells={mc}: nz={m.shape[2]} N={m.n_unknowns} L={v * 1e12:.4f} pH "
              f"({100 * (v / base - 1):+.3f} %)", flush=True)

    out["vertical_grid"] = []
    for bdz in (BASE_DZ, 0.5 * BASE_DZ, 0.25 * BASE_DZ):
        m = build_bar(ell, width, div, base_dz=bdz)
        if m.n_unknowns > N_MAX:
            continue
        v = float(np.real(l_dut(m)))
        out["vertical_grid"].append({"base_dz": bdz, "nz": int(m.shape[2]),
                                     "n_unknowns": int(m.n_unknowns), "L_dut": v,
                                     "rel": v / base - 1.0})
        print(f"    base_dz={bdz * 1e6:.2f} um: nz={m.shape[2]} N={m.n_unknowns} L={v * 1e12:.4f} pH "
              f"({100 * (v / base - 1):+.3f} %)", flush=True)
    return out


def wall_extrapolation_check(sg, ell: float, width: float, div: float,
                             pads: Sequence[int] = (4, 6, 8, 12),
                             n_max: int = N_MAX) -> dict[str, Any]:
    """The error bar on the wall correction itself: the 3-point Richardson in
    ``1 / wall_distance`` built from ``pad_cells`` 4 / 6 / 8 against a
    DIRECT solve with ``pad_cells = 12``, whose walls are ~30 x further out
    than the pad = 4 ones. Everything the conclusion of this study rests on
    goes through that extrapolation, so it is measured, not assumed."""
    rows = []
    for pad in pads:
        r = run_case(sg, ell, width, div, pad=pad, n_max=max(n_max, 80_000),
                     with_pec=False, verbose=False)
        if r is None:
            continue
        rows.append({"pad_cells": pad, "L_dut": r["L_dut"],
                     "wall_distance": r["grid"]["wall_distance"],
                     "n_unknowns": r["grid"]["n_unknowns"], "seconds": r["seconds"]})
        print(f"    pad={pad}: N={rows[-1]['n_unknowns']} wall="
              f"{rows[-1]['wall_distance'] * 1e6:.0f} um  L={rows[-1]['L_dut'] * 1e12:.4f} pH "
              f"({rows[-1]['seconds']:.0f} s)", flush=True)
    out: dict[str, Any] = {"bar_length": ell, "width": width, "div": div, "rows": rows}
    if len(rows) >= 4:
        rich = richardson([1.0 / r["wall_distance"] for r in rows[:3]],
                          [r["L_dut"] for r in rows[:3]])
        est = float(np.mean(rich["range"]))
        out["extrapolated_from_4_6_8"] = est
        out["direct_pad12"] = rows[3]["L_dut"]
        out["extrapolation_error"] = est / rows[3]["L_dut"] - 1.0
        print(f"      extrapolated {est * 1e12:.4f} vs direct pad=12 "
              f"{rows[3]['L_dut'] * 1e12:.4f} pH "
              f"({100 * out['extrapolation_error']:+.3f} %)", flush=True)
    return out


# the spiral study's own fixture, written out here (never imported: that
# study's driver is owned elsewhere and must not be able to move this block)
SPIRAL_FIXTURE = dict(n_turns=2, r_out=52e-6, spacing=10e-6, width=10e-6, lead=15e-6)


def spiral_wall_cross_check(divs: Sequence[float] = (1.0, 2.0), pads: Sequence[int] = WALL_PADS,
                            n_max: int = N_MAX) -> dict[str, Any]:
    """The finding of this study applied back to the SPIRAL: how much of the
    spiral study's own level-to-level change is the PEC walls creeping in?

    The spiral study fixed ``pad_cells = 4`` and gated the walls only at its
    coarsest level (+0.06 % for 4 -> 8 there). Because ``pad_lines`` grows
    from a ``base_dx``-sized cell, the wall distance falls with every
    refinement, so that gate does not carry to W/2 and W/3. Here the spiral
    fixture itself (2 turns, r_out = 52 um, W = S = 10 um, lead 1.5 W, the
    same stack, vacuum dielectrics, sigma_volumetric = 3e6) is rebuilt at
    ``pad_cells`` 4 / 6 / 8 on its W/1 and W/2 grids and ``L_diff`` is
    recorded. Nothing here modifies or imports the spiral study."""
    from rfx.fdfd import spiral as sm
    out: dict[str, Any] = {"fixture": dict(SPIRAL_FIXTURE), "levels": {}}
    for div in divs:
        rows = []
        for pad in pads:
            spec = sm.SpiralSpec(base_dx=SPIRAL_FIXTURE["width"] / div, margin=MARGIN,
                                 stack=stack(), port_gap_cells=1, pad_cells=pad,
                                 pad_ratio=PAD_RATIO, **SPIRAL_FIXTURE)
            m = sm.build_spiral(spec)
            if m.n_unknowns > n_max:
                print(f"    spiral W/{div:g} pad={pad}: N={m.n_unknowns} > {n_max}, SKIPPED",
                      flush=True)
                continue
            t0 = time.time()
            v = float(np.real(sm.solve_spiral(m, FREQ, sigma_volumetric=SIGMA_VOL).L_diff))
            rows.append({"pad_cells": pad, "L_dut": v, "n_unknowns": int(m.n_unknowns),
                         "shape": list(m.shape), "wall_x": float(m.x_nom[-1]),
                         "seconds": time.time() - t0})
            print(f"    spiral W/{div:g} pad={pad}: N={m.n_unknowns} wall_x="
                  f"{rows[-1]['wall_x'] * 1e6:.0f} um  L={v * 1e12:.3f} pH "
                  f"({rows[-1]['seconds']:.0f} s)", flush=True)
        if len(rows) < 2:
            continue
        entry: dict[str, Any] = {"rows": rows, "L_pad4": rows[0]["L_dut"],
                                 "rel_pad4_to_last": rows[-1]["L_dut"] / rows[0]["L_dut"] - 1.0}
        rich = richardson([1.0 / r["wall_x"] for r in rows], [r["L_dut"] for r in rows])
        entry["wall_richardson"] = rich
        entry["L_wall_extrapolated"] = float(np.mean(rich["range"]))
        entry["rel_pad4"] = entry["L_wall_extrapolated"] / entry["L_pad4"] - 1.0
        print(f"      -> wall-extrapolated {entry['L_wall_extrapolated'] * 1e12:.3f} pH "
              f"({100 * entry['rel_pad4']:+.2f} % on pad=4)", flush=True)
        out["levels"][f"{div:g}"] = entry
    return out


DZ_UNDER = 0.5 * T_M1 + T_VIA + 0.5 * T_M2          # 4 um, M2 centre -> M1 centre


def spiral_lead_differential(sg, divs: Sequence[float] = (1.0, 2.0), lead_from: float = 15e-6,
                             lead_to: float = 35e-6, n_max: int = N_MAX) -> dict[str, Any]:
    """The spiral's OWN plane-free differential, at more than one level.

    The spiral study measured it only at ``base_dx = W`` (lengthening the
    lead by 20 um adds 15.42 pH in the FDFD against 20.21 pH in the referee,
    ratio 0.763) and concluded that its gap is distributed along the
    conductor. This study needs the SECOND level to decide the attribution:
    if the spiral's straight runs improve with refinement the way the bar's
    do (0.829 -> 0.932 -> 0.975, wall-extrapolated), then whatever is left
    of the spiral's -14 % at W/3 is NOT the straight runs.

    Lengthening ``lead`` from 1.5 W to 3.5 W adds exactly 20 um to the lead
    AND 20 um to the underpass (which ends on the port line) in both tools;
    every fixture systematic is common to the two solves and cancels. The
    referee difference is taken in the plain Greenhouse corner convention
    because the corners are identical in both terms and cancel exactly (the
    area-exact translation would move both by the same amount), and the
    reference-plane shift cancels for the same reason."""
    from rfx.fdfd import spiral as sm

    def ref(lead: float) -> float:
        segs = sg.rect_spiral_segments(
            SPIRAL_FIXTURE["n_turns"], SPIRAL_FIXTURE["r_out"], SPIRAL_FIXTURE["width"],
            SPIRAL_FIXTURE["spacing"], T_M2, lead_length=lead - 0.5 * SPIRAL_FIXTURE["width"],
            underpass=(DZ_UNDER, T_M1))
        return float(sg.greenhouse_inductance(segs, ground_height=GROUND_H))

    d_ref = ref(lead_to) - ref(lead_from)
    out: dict[str, Any] = {"fixture": dict(SPIRAL_FIXTURE), "lead_from": lead_from,
                           "lead_to": lead_to, "added_conductor_length": 2.0 * (lead_to - lead_from),
                           "L_referee_short_lead": ref(lead_from), "dL_referee": d_ref,
                           "levels": {}}
    for div in divs:
        vals = {}
        for lead in (lead_from, lead_to):
            kw = dict(SPIRAL_FIXTURE)
            kw["lead"] = lead
            spec = sm.SpiralSpec(base_dx=SPIRAL_FIXTURE["width"] / div, margin=MARGIN,
                                 stack=stack(), port_gap_cells=1, pad_cells=PAD_CELLS,
                                 pad_ratio=PAD_RATIO, **kw)
            m = sm.build_spiral(spec)
            if m.n_unknowns > n_max:
                print(f"    spiral W/{div:g} lead={lead * 1e6:.0f} um: N={m.n_unknowns} > {n_max}, "
                      "SKIPPED", flush=True)
                vals = {}
                break
            t0 = time.time()
            vals[lead] = {"L_dut": float(np.real(sm.solve_spiral(
                m, FREQ, sigma_volumetric=SIGMA_VOL).L_diff)),
                "n_unknowns": int(m.n_unknowns), "seconds": time.time() - t0}
        if not vals:
            continue
        d_f = vals[lead_to]["L_dut"] - vals[lead_from]["L_dut"]
        out["levels"][f"{div:g}"] = {
            "div": div, "base_dx": SPIRAL_FIXTURE["width"] / div,
            "L_short_lead": vals[lead_from]["L_dut"], "L_long_lead": vals[lead_to]["L_dut"],
            "n_unknowns": [vals[lead_from]["n_unknowns"], vals[lead_to]["n_unknowns"]],
            "seconds": [vals[lead_from]["seconds"], vals[lead_to]["seconds"]],
            "dL_fdfd": d_f, "ratio_fdfd_over_referee": d_f / d_ref}
        print(f"    spiral W/{div:g}: dL_fdfd={d_f * 1e12:+.3f} dL_ref={d_ref * 1e12:+.3f} pH  "
              f"ratio={d_f / d_ref:.4f}", flush=True)
    return out


def gradient_check(ell: float, width: float, div: float) -> dict[str, Any]:
    """B4: ``jax.value_and_grad`` of the de-embedded ``L_diff`` in
    ``theta = (l_bar, W)`` against a 4th-order central finite difference
    with 1 % steps (which move L by ~1e-2 relative, far above the ~1e-9 LU
    noise floor), on one level of the real study fixture."""
    import jax
    import jax.numpy as jnp

    model = build_bar(ell, width, div)
    theta0 = (ell, width)

    def scalar(t) -> Any:
        return jnp.real(l_dut(model, t))

    t0 = time.time()
    val, grad = jax.value_and_grad(scalar)(jnp.asarray(theta0, dtype=jnp.float64))
    jax.block_until_ready(grad)
    rec: dict[str, Any] = {"bar_length": ell, "width": width, "div": div,
                           "n_unknowns": int(model.n_unknowns),
                           "L_dut": float(val), "grad": [float(v) for v in grad],
                           "value_and_grad_seconds": time.time() - t0,
                           "fd_step_rel": FD_STEP_REL, "params": ["l_bar", "width"]}
    t0 = time.time()
    fd = []
    for k in range(2):
        def f(v: float, k: int = k) -> float:
            t = list(theta0)
            t[k] = v
            return float(scalar(jnp.asarray(t, dtype=jnp.float64)))
        fd.append(float(fd4(f, theta0[k], FD_STEP_REL * theta0[k])))
    rec["fd4"] = fd
    rec["fd_seconds"] = time.time() - t0
    rec["grad_vs_fd_rel"] = [abs(g - f_) / abs(f_) for g, f_ in zip(rec["grad"], fd)]
    print(f"    dL/dl_bar={rec['grad'][0]:.6e} (FD4 {fd[0]:.6e}), "
          f"dL/dW={rec['grad'][1]:.6e} (FD4 {fd[1]:.6e}); worst rel "
          f"{max(rec['grad_vs_fd_rel']):.2e}  ({rec['value_and_grad_seconds']:.0f}+"
          f"{rec['fd_seconds']:.0f} s)", flush=True)
    return rec


# ----------------------------------------------------------------------------
# 7. gates

def evaluate_gates(study: dict[str, Any]) -> dict[str, Any]:
    g: dict[str, Any] = {}
    levels = study["levels"]
    refs = study["referee"]

    # --- B1: per-level gap and the observed order --------------------------
    b1: dict[str, Any] = {"per_case": {}}
    for key in sorted(levels):
        rows = [levels[key][k] for k in sorted(levels[key], key=float)]
        gaps = {f"W/{r['div']:g}": r["gap"] for r in rows}
        shrink = {conv: bool(all(abs(b["gap"][conv]) <= abs(a["gap"][conv]) + 1e-12
                                 for a, b in zip(rows, rows[1:]))) for conv in PLANES}
        b1["per_case"][key] = {
            "referee_centres": refs[key]["L_centres"],
            "L_dut": {f"W/{r['div']:g}": r["L_dut"] for r in rows},
            "n_unknowns": {f"W/{r['div']:g}": r["grid"]["n_unknowns"] for r in rows},
            "seconds": {f"W/{r['div']:g}": r["seconds"] for r in rows},
            "gap": gaps,
            "gap_shrinks_monotonically": shrink,
            "observed_order": study["richardson"][key]["observed_order"],
            "implied_length": {f"W/{r['div']:g}": r["implied_length"] for r in rows},
            "implied_plane_shift": {f"W/{r['div']:g}": r["implied_plane_shift"] for r in rows},
        }
        wc = study.get("wall_corrected_levels", {}).get(key, {})
        if wc:
            b1["per_case"][key]["wall_corrected"] = {
                "L": {f"W/{e['div']:g}": e["L_wall_extrapolated"] for e in wc.values()},
                "rel_pad4": {f"W/{e['div']:g}": e["rel_pad4"] for e in wc.values()},
                "gap": {f"W/{e['div']:g}": e["gap"] for e in wc.values()},
                "implied_length": {f"W/{e['div']:g}": e["implied_length"] for e in wc.values()},
            }
    b1["reported_only"] = True
    g["B1"] = b1

    # --- B2: the differential ratio ----------------------------------------
    b2: dict[str, Any] = {"tolerance": 0.03, "per_width": {}}
    ok = []
    for wk, d in study["differential"].items():
        keys = sorted(d["levels"], key=float)
        if not keys:
            continue
        fin = d["levels"][keys[-1]]
        entry = {
            "finest_div": fin["div"], "finest_base_dx": fin["base_dx"],
            "ratio_per_level": {f"W/{d['levels'][k]['div']:g}": d["levels"][k]["ratio_centres"]
                                for k in keys},
            "ratio_post_edges_per_level": {f"W/{d['levels'][k]['div']:g}":
                                           d["levels"][k]["ratio_post_edges"] for k in keys},
            "ratio_wall_corrected_per_level": {
                f"W/{d['levels'][k]['div']:g}": d["levels"][k]["ratio_centres_wall_corrected"]
                for k in keys if "ratio_centres_wall_corrected" in d["levels"][k]},
            "finest_ratio": fin["ratio_centres"],
            "finest_ratio_wall_corrected": fin.get("ratio_centres_wall_corrected"),
            "finest_deviation": abs(fin["ratio_centres"] - 1.0),
            "passed": bool(abs(fin["ratio_centres"] - 1.0) <= 0.03),
        }
        if entry["finest_ratio_wall_corrected"] is not None:
            entry["finest_deviation_wall_corrected"] = abs(entry["finest_ratio_wall_corrected"] - 1.0)
            entry["passed_wall_corrected"] = bool(entry["finest_deviation_wall_corrected"] <= 0.03)
        ok.append(entry["passed"])
        b2["per_width"][wk] = entry
    b2["passed"] = bool(ok) and bool(all(ok))
    wc = [e["passed_wall_corrected"] for e in b2["per_width"].values()
          if "passed_wall_corrected" in e]
    b2["passed_wall_corrected"] = bool(wc) and bool(all(wc))
    b2["finest_ratio_wall_corrected"] = {
        k: e.get("finest_ratio_wall_corrected") for k, e in b2["per_width"].items()}
    g["B2"] = b2

    # --- B3: Richardson vs the referee -------------------------------------
    b3: dict[str, Any] = {"tolerance": 0.03, "per_case": {}}
    ok = []
    for key in sorted(levels):
        rich = study["richardson"][key]
        rows = [levels[key][k] for k in sorted(levels[key], key=float)]
        fine_dx = rows[-1]["base_dx"]
        ref = {"centres": refs[key]["L_centres"], "inner_edges": refs[key]["L_inner_edges"],
               "post_edges": rows[-1]["L_referee"]["post_edges"]}
        entry: dict[str, Any] = {"estimates": rich["estimates"], "range": rich["range"],
                                 "observed_order": rich["observed_order"],
                                 "finest_base_dx": fine_dx, "referee": ref, "rel_range": {}}
        for conv, v in ref.items():
            entry["rel_range"][conv] = [rich["range"][0] / v - 1.0, rich["range"][1] / v - 1.0]
        rw = study.get("richardson_wall_extrapolated", {}).get(key)
        if rw:
            entry["wall_extrapolated"] = {
                "estimates": rw["estimates"], "range": rw["range"],
                "observed_order": rw["observed_order"],
                "rel_range": {c: [rw["range"][0] / v - 1.0, rw["range"][1] / v - 1.0]
                              for c, v in ref.items()}}
            entry["wall_extrapolated"]["worst_rel_centres"] = max(
                abs(x) for x in entry["wall_extrapolated"]["rel_range"]["centres"])
            entry["wall_extrapolated"]["passed"] = bool(
                entry["wall_extrapolated"]["worst_rel_centres"] <= 0.03)
        wcl = study.get("wall_corrected_levels", {}).get(key, {})
        if wcl:
            # the per-level statement the conclusion rests on: with the walls
            # extrapolated away, how far is L from the referee taken at the
            # plane the short standard actually grounds (post_edges)?
            fine = [e for e in wcl.values() if e["div"] >= 3.0]
            entry["wall_corrected_post_edges_gap"] = {
                f"W/{e['div']:g}": e["gap"]["post_edges"] for e in wcl.values()}
            entry["wall_corrected_within_3pct_post_edges_from_div3"] = bool(
                fine and all(abs(e["gap"]["post_edges"]) <= 0.03 for e in fine))
            entry["wall_corrected_implied_length"] = {
                f"W/{e['div']:g}": e["implied_length"] for e in wcl.values()}
        worst = max(abs(x) for x in entry["rel_range"]["centres"])
        entry["worst_rel_centres"] = worst
        entry["passed"] = bool(worst <= 0.03)
        ok.append(entry["passed"])
        b3["per_case"][key] = entry
    b3["passed"] = bool(ok) and bool(all(ok))
    g["B3"] = b3

    # --- B4: AD vs FD4 ------------------------------------------------------
    gr = study.get("gradient")
    if gr:
        worst = max(gr["grad_vs_fd_rel"])
        g["B4"] = {"params": gr["params"], "grad": gr["grad"], "fd4": gr["fd4"],
                   "grad_vs_fd_rel": gr["grad_vs_fd_rel"], "fd_step_rel": gr["fd_step_rel"],
                   "div": gr["div"], "n_unknowns": gr["n_unknowns"],
                   "worst": worst, "tolerance": 1e-4, "passed": bool(worst <= 1e-4)}

    # --- the area gate is trivial for a bar --------------------------------
    g["area"] = {"note": "the bar's plan area is l * W exactly; no corner convention exists",
                 "checks": study["area_gate"], "passed": study["area_gate"]["passed"]}
    return g


def area_gate(ell: float, width: float) -> dict[str, Any]:
    """The trivial analogue of the spiral study's corner gate: the drawn bar
    polygon is a single rectangle of area ``(l + W) * W`` (the bar plus its
    two end squares) and the DE-EMBEDDED conductor the referee models is the
    concentric ``l * W`` sub-rectangle, both exact to float round-off. There
    is no double-counted corner square anywhere."""
    from rfx.fdfd import gds
    from rfx.fdfd import spiral as sm
    sp = sm.straight_bar_geometry(ell, width)
    polys = sp.polygons[sm.KEY_M2]
    drawn = gds.polygon_area(polys[0])
    feet = [gds.polygon_area(p) for p in polys[1:]]
    out = {
        "n_polygons": len(polys),
        "drawn_area": drawn, "drawn_over_expected_minus_1": drawn / ((ell + width) * width) - 1.0,
        "footprint_areas": feet,
        "footprint_over_expected_minus_1": [a / (width * width) - 1.0 for a in feet],
        "referee_area": ell * width,
        "referee_over_drawn_minus_1": (ell * width) / drawn - 1.0,
    }
    out["passed"] = bool(abs(out["drawn_over_expected_minus_1"]) <= 1e-12
                         and max(abs(v) for v in out["footprint_over_expected_minus_1"]) <= 1e-12)
    return out


# ----------------------------------------------------------------------------
# 8. figure

def figure(study: dict[str, Any], path: pathlib.Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colours = ("#1f77b4", "#d62728", "#2ca02c", "#ff7f0e")
    fig, axes = plt.subplots(2, 2, figsize=(13.6, 9.6))
    ax1, ax2, ax3, ax4 = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]
    wcl = study.get("wall_corrected_levels", {})

    # (a) the raw data: L against the grid, with the referee's plane band
    for c, key in zip(colours, sorted(study["levels"])):
        rows = [study["levels"][key][k] for k in sorted(study["levels"][key], key=float)]
        h = np.array([r["base_dx"] for r in rows]) * 1e6
        ax1.plot(h, np.array([r["L_dut"] for r in rows]) * 1e12, "o-", color=c,
                 label=f"{key} pad_cells = 4")
        wc = sorted(wcl.get(key, {}).values(), key=lambda e: -e["base_dx"])
        if wc:
            ax1.plot([e["base_dx"] * 1e6 for e in wc],
                     [e["L_wall_extrapolated"] * 1e12 for e in wc], "s--", color=c, alpha=0.75,
                     label=f"{key} wall-extrapolated")
        ref = study["referee"][key]
        ax1.axhline(ref["L_centres"] * 1e12, color=c, linestyle="-", linewidth=1.0, alpha=0.6)
        ax1.axhline(ref["L_inner_edges"] * 1e12, color=c, linestyle=":", linewidth=1.0, alpha=0.6)
    ax1.set_xlabel("base_dx [um]")
    ax1.set_ylabel("L [pH]")
    ax1.set_xlim(0.0, None)
    ax1.set_title("(a) de-embedded L vs in-plane refinement\n"
                  "horizontal: referee at plane 'centres' (solid) and 'inner_edges' (dotted)")
    ax1.legend(fontsize=6, ncol=2)
    ax1.grid(alpha=0.3)

    # (b) the gap against the plane the SHORT standard actually grounds
    for c, key in zip(colours, sorted(study["levels"])):
        rows = [study["levels"][key][k] for k in sorted(study["levels"][key], key=float)]
        ax2.plot(np.array([r["base_dx"] for r in rows]) * 1e6,
                 [100.0 * r["gap"]["post_edges"] for r in rows], "o-", color=c,
                 label=f"{key} pad_cells = 4")
        wc = sorted(wcl.get(key, {}).values(), key=lambda e: -e["base_dx"])
        if wc:
            ax2.plot([e["base_dx"] * 1e6 for e in wc],
                     [100.0 * e["gap"]["post_edges"] for e in wc], "s--", color=c, alpha=0.75,
                     label=f"{key} wall-extrapolated")
    ax2.axhline(0.0, color="k", linewidth=1)
    ax2.axhspan(-3.0, 3.0, color="k", alpha=0.08, label="+-3 %")
    ax2.set_xlabel("base_dx [um]")
    ax2.set_ylabel("L_FDFD / L_referee - 1 [%]")
    ax2.set_xlim(0.0, None)
    ax2.set_ylim(-30.0, 15.0)
    ax2.set_title("(b) gap against the referee at the plane the short standard grounds\n"
                  "l_eff = l - W - 2 base_dx")
    ax2.legend(fontsize=6, ncol=2)
    ax2.grid(alpha=0.3)

    # (c) the plane-free differential -- the measurement this study exists for
    for c, (wk, d) in zip(colours, sorted(study["differential"].items())):
        keys = sorted(d["levels"], key=float)
        ax3.plot([d["levels"][k]["base_dx"] * 1e6 for k in keys],
                 [d["levels"][k]["ratio_centres"] for k in keys], "o-", color=c,
                 label=f"bar {wk} raw (pad_cells = 4)")
        rc = [(d["levels"][k]["base_dx"] * 1e6, d["levels"][k]["ratio_centres_wall_corrected"])
              for k in keys if "ratio_centres_wall_corrected" in d["levels"][k]]
        if rc:
            ax3.plot([a for a, _ in rc], [b for _, b in rc], "s--", color=c, alpha=0.75,
                     label=f"bar {wk} wall-corrected")
    sld = study.get("spiral_lead_differential")
    if sld and sld["levels"]:
        keys = sorted(sld["levels"], key=float)
        ax3.plot([sld["levels"][k]["base_dx"] * 1e6 for k in keys],
                 [sld["levels"][k]["ratio_fdfd_over_referee"] for k in keys], "^-.",
                 color="#8c564b", linewidth=1.6, markersize=9,
                 label="SPIRAL lead differential, same protocol")
    ax3.axhline(1.0, color="k", linewidth=1)
    ax3.axhspan(0.97, 1.03, color="k", alpha=0.08, label="B2 +-3 %")
    ax3.set_xlabel("base_dx [um]")
    ax3.set_ylabel("[dL]_FDFD / [dL]_referee")
    ax3.set_xlim(0.0, None)
    ax3.set_title("(c) B2: the plane-free differential [L(l2) - L(l1)]\n"
                  "the bar and the spiral lose the same inductance per unit length")
    ax3.legend(fontsize=6)
    ax3.grid(alpha=0.3)

    # (d) the wall bias, and the same ladder on the spiral
    wl = study.get("wall_ladder", {})
    for c, key in zip(colours, sorted(wl)):
        for i, (dkey, entry) in enumerate(sorted(wl[key].items(), key=lambda kv: float(kv[0]))):
            d_ = np.array([r["wall_distance"] for r in entry["rows"]]) * 1e6
            v_ = np.array([r["L_dut"] for r in entry["rows"]])
            ax4.plot(d_, 100.0 * (v_ / entry["L_pad4"] - 1.0), "o-", color=c, alpha=0.8,
                     label=f"bar {key}" if i == 0 else None)
    sp = study.get("spiral_wall_cross_check", {}).get("levels", {})
    for dkey, entry in sorted(sp.items(), key=lambda kv: float(kv[0])):
        d_ = np.array([r["wall_x"] for r in entry["rows"]]) * 1e6
        v_ = np.array([r["L_dut"] for r in entry["rows"]])
        ax4.plot(d_, 100.0 * (v_ / entry["L_pad4"] - 1.0), "^-.", color="#8c564b", linewidth=1.6,
                 markersize=9, label=f"SPIRAL W/{float(dkey):g}")
    ax4.set_xscale("log")
    ax4.axhline(0.0, color="k", linewidth=1)
    ax4.set_xlabel("distance from the conductor to the nearest PEC wall [um]")
    ax4.set_ylabel("L relative to pad_cells = 4 [%]")
    ax4.set_title("(d) the wall bias of pad_cells = 4 grows with refinement for an OPEN bar\n"
                  "and is negligible for the closed spiral loop")
    ax4.legend(fontsize=6, ncol=2)
    ax4.grid(alpha=0.3)

    fig.suptitle("D2b: straight bar on the spiral fixture vs the Greenhouse referee "
                 f"({FREQ / 1e6:.0f} MHz, uniform-current metal)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------
# 9. main

def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--widths", type=float, nargs="+", default=[w * 1e6 for w in WIDTHS],
                    help="strip widths in um")
    ap.add_argument("--lengths", type=float, nargs="+", default=[v * 1e6 for v in LENGTHS],
                    help="bar lengths in um (l2 = 2 l1 expected)")
    ap.add_argument("--levels", type=float, nargs="+", default=list(LEVELS),
                    help="base_dx = width / level")
    ap.add_argument("--n-max", type=int, default=N_MAX)
    ap.add_argument("--probe-div", type=float, default=3.0,
                    help="level used for the systematics and the B4 gradient check")
    ap.add_argument("--no-walls", action="store_true")
    ap.add_argument("--no-systematics", action="store_true")
    ap.add_argument("--no-spiral-cross-check", action="store_true")
    ap.add_argument("--from-json", action="store_true",
                    help="reuse the per-level records already in the JSON")
    args = ap.parse_args(argv)

    import jax
    jax.config.update("jax_enable_x64", True)
    sg = load_referee()

    widths = [w * 1e-6 for w in args.widths]
    lengths = [v * 1e-6 for v in args.lengths]
    t_start = time.time()
    previous: dict[str, Any] = json.loads(JSON_PATH.read_text()) if (
        args.from_json and JSON_PATH.exists()) else {}

    study: dict[str, Any] = {
        "fixture": {
            "freq": FREQ, "widths": widths, "lengths": lengths, "levels": list(args.levels),
            "margin": MARGIN, "base_dz": BASE_DZ, "metal_cells": METAL_CELLS,
            "pad_cells": PAD_CELLS, "pad_ratio": PAD_RATIO, "sigma_volumetric": SIGMA_VOL,
            "ground_height": GROUND_H, "port_gap_cells": 1, "eps_dielectrics": 1.0,
            "n_max": args.n_max, "fd_step_rel": FD_STEP_REL,
            "stack": {"t_si": T_SI, "t_ox_low": T_OX_LOW, "t_m1": T_M1, "t_via": T_VIA,
                      "t_m2": T_M2, "t_ox_high": T_OX_HIGH, "t_air": T_AIR},
        },
        # the spiral study's published numbers, for the comparison this study exists to make
        "spiral_reference": {
            "note": "QUOTED from the spiral study's JSON for comparison; NOT measured here. "
                    "Everything measured in this session is in spiral_wall_cross_check and "
                    "spiral_lead_differential, which rebuild that fixture from its published "
                    "parameters (the W/1 pad=4 L_diff reproduces to 262.439 pH).",
            "source": "validation/fdfd/spiral_convergence.json",
            "referee_area_exact": 349.37e-12, "L_dut_W1": 262.44e-12,
            "gap_W1": -0.2488, "gap_W2": -0.1637, "gap_W3": -0.1420,
            "observed_order": 1.48, "richardson_range_rel": [-0.124, -0.099],
            "lead_differential_ratio_W1": 0.7632,
            "reference_plane_sensitivity": [0.0146, -0.0146],
        },
        "referee": {}, "levels": dict(previous.get("levels", {})),
        "area_gate": {},
    }

    def dump() -> None:
        study["seconds"] = time.time() - t_start
        JSON_PATH.write_text(json.dumps(study, indent=1, sort_keys=False))

    # --- referee (host numpy, instant) -------------------------------------
    print("referee (Greenhouse self + PEC-ground image of one bar):", flush=True)
    for w in widths:
        for ell in lengths:
            key = case_key(w, ell)
            r = referee_block(sg, ell, w)
            study["referee"][key] = r
            print(f"  {key}: L={r['L_centres'] * 1e12:8.3f} pH  self={r['self_sum'] * 1e12:8.3f} "
                  f"image={r['image_sum'] * 1e12:+8.3f}  free-space={r['L_free_space'] * 1e12:8.3f} "
                  f"(shielding {100 * r['ground_shielding']:+.1f} %)  plane band "
                  f"[{100 * r['reference_plane_sensitivity'][0]:+.2f}, "
                  f"{100 * r['reference_plane_sensitivity'][1]:+.2f}] %  "
                  f"worst_err={r['worst_error']:.1e}", flush=True)
    study["area_gate"] = area_gate(lengths[0], widths[0])
    print(f"  area gate: drawn/expected-1={study['area_gate']['drawn_over_expected_minus_1']:+.2e} "
          f"passed={study['area_gate']['passed']}", flush=True)
    dump()

    # --- levels -------------------------------------------------------------
    print("FDFD levels (volumetric metal, pad_cells = 4 graded wall padding):", flush=True)
    for w in widths:
        for ell in lengths:
            key = case_key(w, ell)
            print(f"  {key}:", flush=True)
            study["levels"].setdefault(key, {})
            # reused records register their grid too, so a level whose grid is
            # identical to an already-recorded one is dropped on a resume as
            # well (two identical values at different nominal h would break
            # the Richardson fit)
            seen_shapes: dict[tuple, float] = {
                tuple(r["grid"]["shape"]): r["div"] for r in study["levels"][key].values()}
            for div in args.levels:
                dk = f"{div:g}"
                if dk in study["levels"][key]:
                    print(f"    W/{div:g}: reused from the JSON "
                          f"(L={study['levels'][key][dk]['L_dut'] * 1e12:.3f} pH)", flush=True)
                    continue
                m = build_bar(ell, w, div)
                shp = tuple(m.shape)
                if shp in seen_shapes:
                    print(f"    W/{div:g}: grid {shp} identical to W/{seen_shapes[shp]:g}, dropped",
                          flush=True)
                    continue
                seen_shapes[shp] = div
                rec = run_case(sg, ell, w, div, n_max=args.n_max,
                               with_pec=len(study["levels"][key]) < 2)
                if rec is None:
                    continue
                study["levels"][key][dk] = rec
                dump()

    # --- Richardson ---------------------------------------------------------
    study["richardson"] = {}
    for key, rows in study["levels"].items():
        lv = [rows[k] for k in sorted(rows, key=float)]
        study["richardson"][key] = richardson([r["base_dx"] for r in lv], [r["L_dut"] for r in lv])
    dump()

    # --- wall ladder --------------------------------------------------------
    study["wall_ladder"] = dict(previous.get("wall_ladder", {})) if args.from_json else {}
    if not args.no_walls:
        print("wall ladder (pad_cells 4 / 6 / 8 at every level):", flush=True)
        for w in widths:
            for ell in lengths:
                key = case_key(w, ell)
                if key in study["wall_ladder"]:
                    print(f"  {key}: reused from the JSON", flush=True)
                    continue
                print(f"  {key}:", flush=True)
                divs = [study["levels"][key][k]["div"]
                        for k in sorted(study["levels"][key], key=float)]
                study["wall_ladder"][key] = wall_ladder(sg, ell, w, divs, n_max=args.n_max,
                                                       known=study["levels"][key])
                dump()

    # --- the wall-extrapolated series ---------------------------------------
    # only ladders with THREE pad_cells points feed this series: the 3-point
    # Richardson in 1/wall_distance was validated against a direct pad = 12
    # solve (+0.026 %, ``wall_extrapolation_check``), a 2-point one was not
    # and measurably overshoots (pad 4/6 alone gives +10.4 % at W10_l100 W/4
    # against +5.5 % for 4/6/8 one level coarser).
    study["richardson_wall_extrapolated"] = {}
    for key, ladder in study["wall_ladder"].items():
        rows = [(study["levels"][key][k]["base_dx"], ladder[k]["L_wall_extrapolated"],
                 len(ladder[k]["rows"])) for k in sorted(ladder, key=float)
                if "L_wall_extrapolated" in ladder[k]]
        keep = [(h, v) for h, v, n in rows if n >= 3]
        if len(keep) >= 2:
            r = richardson([h for h, _ in keep], [v for _, v in keep])
            r["n_pad_points"] = [n for _, _, n in rows]
            r["levels_used"] = len(keep)
            study["richardson_wall_extrapolated"][key] = r
    # per-level wall-corrected L and its gap against all three plane conventions
    study["wall_corrected_levels"] = {}
    for key, ladder in study["wall_ladder"].items():
        w = study["referee"][key]["width"]
        out: dict[str, Any] = {}
        for k in sorted(ladder, key=float):
            e = ladder[k]
            if "L_wall_extrapolated" not in e or len(e["rows"]) < 3:
                continue
            lv = study["levels"][key][k]
            v = e["L_wall_extrapolated"]
            out[k] = {"div": lv["div"], "base_dx": lv["base_dx"], "L_wall_extrapolated": v,
                      "rel_pad4": e["rel_pad4"],
                      "gap": {c: v / lv["L_referee"][c] - 1.0 for c in PLANES},
                      "implied_length": invert_referee(sg, v, w)}
        study["wall_corrected_levels"][key] = out
    dump()

    # --- the differential ---------------------------------------------------
    print("differential [L(l2) - L(l1)] FDFD / referee:", flush=True)
    study["differential"] = {}
    if len(lengths) >= 2:
        for w in widths:
            print(f"  W = {w * 1e6:g} um:", flush=True)
            study["differential"][f"W{w * 1e6:g}"] = differential(
                sg, study["levels"], w, lengths[0], lengths[1], study.get("wall_ladder"))
            dump()

    # --- systematics and the gradient ---------------------------------------
    if not args.no_systematics:
        w, ell = max(widths), lengths[0]
        div = args.probe_div
        print(f"sigma plateau ({case_key(w, ell)}, W/{div:g}):", flush=True)
        if args.from_json and "sigma_plateau" in previous:
            study["sigma_plateau"] = previous["sigma_plateau"]
            print("  reused from the JSON", flush=True)
        else:
            study["sigma_plateau"] = sigma_plateau(ell, w, div)
        dump()
        print(f"systematics ({case_key(w, ell)}, W/{div:g}):", flush=True)
        if args.from_json and "systematics" in previous:
            study["systematics"] = previous["systematics"]
            print("  reused from the JSON", flush=True)
        else:
            study["systematics"] = systematics(sg, ell, w, div)
        dump()
        print(f"gradient check B4 ({case_key(w, ell)}, W/{div:g}):", flush=True)
        if args.from_json and "gradient" in previous:
            study["gradient"] = previous["gradient"]
            print("  reused from the JSON", flush=True)
        else:
            study["gradient"] = gradient_check(ell, w, div)
        dump()

    if not args.no_walls:
        print("wall-extrapolation error bar (pad 4/6/8 Richardson vs a direct pad 12):", flush=True)
        if args.from_json and "wall_extrapolation_check" in previous:
            study["wall_extrapolation_check"] = previous["wall_extrapolation_check"]
            print("  reused from the JSON", flush=True)
        else:
            study["wall_extrapolation_check"] = wall_extrapolation_check(
                sg, lengths[0], min(widths), 1.0, n_max=args.n_max)
        dump()

    if not args.no_spiral_cross_check:
        print("spiral wall cross-check (the same wall creep, on the spiral itself):", flush=True)
        if args.from_json and "spiral_wall_cross_check" in previous:
            study["spiral_wall_cross_check"] = previous["spiral_wall_cross_check"]
            print("  reused from the JSON", flush=True)
        else:
            study["spiral_wall_cross_check"] = spiral_wall_cross_check(n_max=args.n_max)
        dump()

    if not args.no_spiral_cross_check:
        print("spiral lead-length differential per level (the attribution):", flush=True)
        if args.from_json and "spiral_lead_differential" in previous:
            study["spiral_lead_differential"] = previous["spiral_lead_differential"]
            print("  reused from the JSON", flush=True)
        else:
            study["spiral_lead_differential"] = spiral_lead_differential(sg, n_max=args.n_max)
        dump()

    study["gates"] = evaluate_gates(study)
    dump()
    figure(study, PNG_PATH)
    print(f"\nwrote {JSON_PATH} and {PNG_PATH} ({study['seconds'] / 60:.1f} min)", flush=True)
    for name, gate in study["gates"].items():
        print(f"  {name}: passed={gate.get('passed')}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
