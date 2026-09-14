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
   3e5 is still 4 % low (there the metal resistance is large enough for the
   shunt capacitance to eat the reactance: Re Z11 = 20.7 Ohm at 3e5 vs
   2.1 Ohm at 3e6). The dielectrics are set to VACUUM (``eps_si = eps_ox =
   1``) because the referee is magnetoquasistatic and has none, and because
   they are what makes that RC contamination large: with the real stack the
   same sigma = 3e5 point sits 18 % low instead of 4 %.
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
     4 vertical cells. L_dut converges monotonically: 190.3 (closed box) /
     244.6 / 245.2 / 245.3 / 245.3 / 245.3 pH for pad = 0 / 3 / 4 / 5 / 6 /
     7 at W/1, so the V5 gate (add 4 cells -> < 1 %) passes with +0.06 %
     and the residual wall systematic at pad = 4 is +0.06 %. The ground
     plane (z = 0) is never padded and never PML: it is the physical wall
     both tools model, and it extends under the padding, i.e. it is
     effectively infinite.

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
   Sensitivity: the port-line and far-edge choices give 354.46 and
   344.27 pH, i.e. +1.46 % / -1.46 % around the 349.37 pH used.

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
``base_dz = 10 um`` and ``metal_cells = 2`` at EVERY level, so the metal
cross-section is always 2 cells across the thickness and the vertical grid
is identical -- the study refines in plane only and the (fixed) vertical
error is reported separately. N = 23062 / 56408 / 108898 unknowns, one
three-fixture solve 15 / 91 / 305 s (SuperLU/COLAMD, x64 CPU). ``W/1.5``
is not a level: the mandatory grid lines of this geometry are 10 um apart,
so ``base_dx`` in (5, 10] um gives the same 2-cell grid as W/2 (measured:
(32,34,15) vs (33,35,15)). ``W/4`` would be 162k unknowns, above the ~105k
single-solve guideline, and is NOT run.

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
V1  FAILS. Uniform-current FDFD ``L_dut`` = 262.44 / 292.20 / 299.77 pH at
    W/1 / W/2 / W/3 against the area-exact referee 349.37 pH: -24.9 %,
    -16.4 %, -14.2 %. Richardson over the three levels gives an observed
    order 1.48 and extrapolates to 314.9 (p = 1, finest pair) / 305.8
    (p = 2) / 308.8 pH (observed order), i.e. the range [305.8, 314.9] pH
    sits -12.4 % to -9.9 % from the referee, outside +-5 %.
    The remaining model difference is the FDFD's DISCRETE SELF-INDUCTANCE
    of a conductor resolved by 1-3 cells across its cross-section, measured
    directly and fixture-free by the lead-length differential: lengthening
    the lead by 20 um adds 7.70 pH in the FDFD against 10.15 pH in the
    referee at W/1 (ratio 0.759) and RATIO at W/2 (see the JSON) -- the
    inductance per unit length of the strip itself is low by about as much
    as L is, so the gap is distributed along the conductor, not a constant
    fixture offset (and the V3 gradient ratios say the same). A strip's
    self-inductance goes as ``ln(2 l / GMD)`` with ``GMD = 0.2235 (W + t)``
    = 2.68 um here; the Yee grid replaces that GMD by one set by the cell
    size, and only a cross-section resolved by many cells recovers it. The
    fixture-side candidates were measured and are all too small to matter:
    de-embedding invariance (``port_gap_cells`` 1 -> 2) 0.03 %, wall
    residual 0.06 %, frequency flatness 1e-6 between 50 and 200 MHz,
    reference plane +-1.46 %, corner convention +1.21 %, vertical grid and
    ``metal_cells`` (see ``systematics`` in the JSON).
V1b L_PEC < L_uniform at every level (the missing DC internal inductance)
    and the gap is reported per level; at W/1 244.88 vs 262.44 pH = -6.7 %,
    against the -7.2 % the referee's own surface-shell variant predicted in
    the first study.
V2  ``jax.grad`` of the volumetric FDFD ``L_dut`` vs FD4 with 1 % steps
    (which move L by ~1e-2 relative, far above the ~1e-9 LU noise floor)
    <= 1e-4 relative, all three parameters, every level.
V3  ``jax.grad`` (FDFD) vs FD4 of the area-exact referee: same sign for all
    three parameters and within 15 % at the finest level run.
V4  the per-level change of ``dL/dwidth`` decreases with refinement.
V5  wall independence: ``pad_cells`` 4 -> 8 moves ``L_dut`` by < 1 % at the
    coarsest level (measured +0.06 %).

Scope fence. One geometry, one frequency; in-plane refinement only (the
vertical grid is held fixed and its error is reported, not extrapolated);
the referee is magnetoquasistatic with uniform current density, no
substrate eddy currents and no via. Nothing here validates losses, the
frequency dependence or the PEC metal model -- the PEC numbers are reported
for contrast only. Sizes are CPU-SuperLU sizes.
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
                     lead: float = LEAD, shift: float = REF_SHIFT) -> list:
    """Referee segment list for ``theta = (r_out, spacing, width)``: the
    strip (lead shortened to the de-embedded reference plane) plus the
    underpass. ``convention`` is ``"greenhouse"`` (the delivered
    centreline-to-corner-point convention, (i) in the module doc) or
    ``"area_exact"`` ((ii), the one the gates use)."""
    r_out, spacing, width = (float(v) for v in theta)
    segs = sg.rect_spiral_segments(N_TURNS, r_out, width, spacing, T_M2,
                                   lead_length=lead - shift, underpass=(DZ_UNDER, T_M1))
    if convention == "greenhouse":
        return segs
    if convention != "area_exact":
        raise ValueError(f"unknown corner convention {convention!r}")
    # the underpass is a lone bar on the other layer: no corner, keep it
    return area_exact_chain(sg, segs[:-1]) + [segs[-1]]


def referee_value(sg, theta: Sequence[float], convention: str = "area_exact",
                  lead: float = LEAD, shift: float = REF_SHIFT,
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


def area_gate(sg, theta: Sequence[float], lead: float = LEAD, shift: float = REF_SHIFT) -> dict[str, Any]:
    """Gate C: for both corner conventions, the sum of the bar rectangle
    areas and the area of their UNION against ``gds.polygon_area`` of the
    ``rect_spiral`` strip polygon (built with the same shortened lead) and
    of the underpass polygon. The area-exact convention must match both to
    1e-12 relative."""
    from shapely.geometry import box
    from shapely.ops import unary_union

    from rfx.fdfd import gds

    r_out, spacing, width = (float(v) for v in theta)
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


def referee_gradient(sg, theta: Sequence[float] = THETA0, convention: str = "area_exact") -> list[float]:
    """``dL/dtheta`` of the referee by FD4 with 1 % steps (host numpy, no
    solver noise: the only error is the FD truncation)."""
    base = np.asarray(theta, dtype=np.float64)
    out = []
    for k in range(3):
        def f(v: float, k: int = k) -> float:
            t = base.copy()
            t[k] = v
            return float(referee_value(sg, t, convention).total)
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
# 3. systematics of the FDFD fixture (all on the coarsest grid unless said)

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
# 4. Richardson extrapolation

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


# ----------------------------------------------------------------------------
# 5. gates

def evaluate_gates(study: dict[str, Any]) -> dict[str, Any]:
    ref = study["referee"]["area_exact"]["total"]
    levels = [study["levels"][k] for k in sorted(study["levels"], key=float)]
    g: dict[str, Any] = {}

    rich = study["richardson"]
    rel = [v / ref - 1.0 for v in rich["range"]]
    g["V1"] = {
        "referee_area_exact": ref,
        "per_level_gap": {f"W/{lv['div']:g}": lv["L_dut"] / ref - 1.0 for lv in levels},
        "L_extrapolated_estimates": rich["estimates"],
        "L_extrapolated_range": rich["range"],
        "observed_order": rich["observed_order"],
        "rel_range": rel,
        "tolerance": 0.05,
        "passed": bool(max(abs(rel[0]), abs(rel[1])) <= 0.05),
    }
    gaps = [lv["L_pec_over_L_uniform_minus_1"] for lv in levels]
    g["V1b"] = {
        "per_level": {f"W/{lv['div']:g}": lv["L_pec_over_L_uniform_minus_1"] for lv in levels},
        "all_negative": bool(all(v < 0 for v in gaps)),
        "shrinks_with_refinement": bool(all(abs(b) <= abs(a) for a, b in zip(gaps, gaps[1:]))),
        "within_6_percent": bool(max(abs(v) for v in gaps) <= 0.06),
        "passed": bool(all(v < 0 for v in gaps)
                       and (all(abs(b) <= abs(a) for a, b in zip(gaps, gaps[1:]))
                            or max(abs(v) for v in gaps) <= 0.06)),
    }
    worst = max(max(lv["grad_vs_fd_rel"]) for lv in levels)
    g["V2"] = {"per_level": {f"W/{lv['div']:g}": lv["grad_vs_fd_rel"] for lv in levels},
               "fd_order": {f"W/{lv['div']:g}": lv["fd_order"] for lv in levels},
               "worst": worst, "tolerance": 1e-4, "passed": bool(worst <= 1e-4)}
    gref = study["referee"]["gradient_area_exact_fd4"]
    per = {}
    for lv in levels:
        per[f"W/{lv['div']:g}"] = {
            "ratio": [a / b for a, b in zip(lv["grad"], gref)],
            "same_sign": bool(all(a * b > 0 for a, b in zip(lv["grad"], gref))),
        }
    fin = per[f"W/{levels[-1]['div']:g}"]
    g["V3"] = {"referee_gradient": gref, "per_level": per,
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
    a = study["area_gate"]["area_exact"]
    g["C_area"] = {"sum_over_polygon_minus_1": a["sum_over_polygon_minus_1"],
                   "union_over_sum_minus_1": a["union_over_sum_minus_1"],
                   "tolerance": 1e-12,
                   "passed": bool(abs(a["sum_over_polygon_minus_1"]) <= 1e-12
                                  and abs(a["union_over_sum_minus_1"]) <= 1e-12)}
    return g


# ----------------------------------------------------------------------------
# 6. figure

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

    ax1.plot(h, ldut, "o-", color="#1f77b4", label="FDFD L_dut, uniform current")
    ax1.plot(h, lpec, "s--", color="#9ecae1", label="FDFD L_dut, PEC metal")
    rich = study["richardson"]
    for name, val in rich["estimates"].items():
        ax1.plot([0.0], [val * 1e12], marker="*", markersize=11, linestyle="none", color="#1f77b4")
        ax1.annotate(f"{name} {val * 1e12:.1f}", (0.0, val * 1e12), textcoords="offset points",
                     xytext=(6, -3), fontsize=7, color="#1f77b4")
    lo, hi = rich["range"]
    ax1.fill_between([0.0, h.max() * 1.05], lo * 1e12, hi * 1e12, color="#1f77b4", alpha=0.12,
                     label="Richardson range (p=1, p=2, observed order)")
    ax1.axhline(ref["area_exact"]["total"] * 1e12, color="#d62728", linestyle="-", linewidth=1.4,
                label=f"referee, area-exact corners {ref['area_exact']['total'] * 1e12:.1f} pH")
    ax1.axhline(ref["greenhouse"]["total"] * 1e12, color="#ff7f0e", linestyle="--", linewidth=1,
                label=f"referee, Greenhouse corners {ref['greenhouse']['total'] * 1e12:.1f} pH")
    rp = ref["reference_plane"]
    ax1.fill_between([0.0, h.max() * 1.05], rp["port_line"] * 1e12, rp["far_edge"] * 1e12,
                     color="#d62728", alpha=0.12, label="referee reference-plane band")
    ax1.set_xlabel("base_dx [um]  (W/1, W/2, W/3)")
    ax1.set_ylabel("L [pH]")
    ax1.set_xlim(0.0, h.max() * 1.05)
    ax1.set_title(f"de-embedded L vs in-plane refinement ({FREQ / 1e6:.0f} MHz)")
    ax1.legend(fontsize=7, loc="lower right")
    ax1.grid(alpha=0.3)

    labels = [f"W/{lv['div']:g}" for lv in levels]
    gref = np.array(ref["gradient_area_exact_fd4"])
    grad = np.array([lv["grad"] for lv in levels])
    fdv = np.array([lv["fd"] for lv in levels])
    for k, (name, c) in enumerate(zip(PARAMS, ("#1f77b4", "#2ca02c", "#d62728"))):
        ax2.plot(labels, grad[:, k] / gref[k], "o-", color=c, label=f"dL/d{name}  FDFD / referee")
        ax2.plot(labels, fdv[:, k] / gref[k], "x", color=c, markersize=9,
                 label=f"dL/d{name}  FD / referee")
    ax2.axhline(1.0, color="k", linewidth=1)
    ax2.axhspan(0.85, 1.15, color="k", alpha=0.08, label="V3 +-15 %")
    ax2.set_ylabel("gradient ratio FDFD / referee")
    ax2.set_title("shape derivatives (jax.grad, FD, referee FD4)")
    ax2.legend(fontsize=7)
    ax2.grid(alpha=0.3)

    if ld:
        ax3 = axes[2]
        keys = sorted(ld["levels"], key=float)
        hs = np.array([ld["levels"][k]["base_dx"] for k in keys]) * 1e6
        ratio = np.array([ld["levels"][k]["ratio_fdfd_over_referee"] for k in keys])
        ax3.plot(hs, 100.0 * (1.0 - ratio), "o-", color="#8c564b",
                 label="deficit of dL/d(lead): 1 - FDFD/referee")
        gap = np.array([lv["L_dut"] for lv in levels]) / ref["area_exact"]["total"] - 1.0
        ax3.plot(h, -100.0 * gap, "s--", color="#1f77b4", label="deficit of L_dut vs the referee")
        ax3.plot(hs, 100.0 * (1.0 - ratio[-1]) * hs / hs[-1], ":", color="#8c564b", linewidth=1,
                 label="O(h) through the finest point")
        ax3.set_xlabel("base_dx [um]")
        ax3.set_ylabel("deficit [%]")
        ax3.set_xlim(0.0, h.max() * 1.05)
        ax3.set_ylim(0.0, None)
        ax3.set_title("V1 diagnosis: the deficit is distributed along the strip")
        ax3.legend(fontsize=7)
        ax3.grid(alpha=0.3)

    fig.suptitle("D2: FDFD spiral vs Greenhouse referee, uniform current, walls removed", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------
# 7. main

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
    study["referee"]["gradient_area_exact_fd4"] = referee_gradient(sg, THETA0, "area_exact")
    study["referee"]["gradient_greenhouse_fd4"] = referee_gradient(sg, THETA0, "greenhouse")
    study["area_gate"] = area_gate(sg, THETA0)
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
        print("sigma plateau (A):", flush=True)
        block("sigma_plateau", lambda: sigma_plateau(coarse))
        print("fixture systematics:", flush=True)
        block("systematics", lambda: fixture_systematics(sg, coarse))
        print("lead-length differential per level (the V1 diagnosis):", flush=True)
        block("lead_differential", lambda: lead_differential(
            sg, args.levels, {k: v["L_dut"] for k, v in study["levels"].items()}))

    levels = [study["levels"][k] for k in sorted(study["levels"], key=float)]
    study["richardson"] = richardson([lv["base_dx"] for lv in levels], [lv["L_dut"] for lv in levels])
    study["gates"] = evaluate_gates(study)
    dump()

    figure(study, PNG_PATH)
    print(f"\nwrote {JSON_PATH} and {PNG_PATH} ({study['seconds'] / 60:.1f} min)", flush=True)
    for name, gate in study["gates"].items():
        print(f"  {name}: passed={gate.get('passed')}", flush=True)
    v1 = study["gates"]["V1"]
    print(f"  V1 detail: referee {v1['referee_area_exact'] * 1e12:.2f} pH, extrapolated "
          f"[{v1['L_extrapolated_range'][0] * 1e12:.2f}, {v1['L_extrapolated_range'][1] * 1e12:.2f}] pH "
          f"= [{100 * v1['rel_range'][0]:+.2f}, {100 * v1['rel_range'][1]:+.2f}] %, "
          f"observed order {v1['observed_order']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
