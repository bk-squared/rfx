"""Validation study: the 3-D FDFD spiral inductor (``rfx.fdfd.spiral``)
against the independent Greenhouse referee
(``validation/crossval/comparators/spiral_greenhouse.py``) under grid
refinement, and the shape derivatives three ways.

Run::

    .venv/bin/python validation/fdfd/spiral_convergence.py [--levels 1 2 3 4] [--K 8]

writes ``spiral_convergence.json`` and ``spiral_convergence.png`` next to
this file. Measured on a laptop (18-core Apple silicon, SuperLU/COLAMD,
one three-fixture LU per solve): W/1 4598 unknowns 1.2 s per solve, W/2
20572 5.2 s, W/3 56358 ~40 s, W/4 105212 ~150 s (value_and_grad 232 s);
each level needs 14 solves (value_and_grad, jit, 12 for the FD4 stencil),
so the four-level run takes ~1.4 h and ``--levels 1 2`` ~2.5 min. The W/4
level is 5 % over the ~100k single-solve guideline; it is kept because the
levels must share the same box and the same h-scaling for Richardson.
``--levels-from`` reuses per-level records written by an earlier run.

Geometry (identical in both tools)
----------------------------------
A 2-turn square spiral, ``r_out = 52 um``, ``W = S = 10 um``, lead
``1.5 W``, M2 strip ``2 um`` thick with an M1 underpass ``4 um`` below its
centre, in the FDFD's closed PEC box: ground at ``h = 26 um`` below the
strip centre (20 um silicon, 1 um oxide, M1, via), lid ``19 um`` above it,
side walls ``10 um`` outside the outermost strip edge. PEC metal, lossless
dielectrics, 100 MHz (quasi-static: the D1 study measured L flat to 1e-5
between 50 and 200 MHz). The referee has no dielectrics (permittivity does
not enter a magnetoquasistatic inductance) and no lead columns: the FDFD's
open/short DE-EMBEDDED ``L_diff`` is compared, so the referee's lead ends
at the centre of the FDFD column footprint (``lead_length = lead - W/2``;
the sensitivity to that choice, +1.8 % for a lead ending at the port line,
is reported).

The referee and the box
-----------------------
The delivered referee models a PEC GROUND only (one image). The FDFD box
also has a PEC lid and four PEC side walls, and at this size they are not
a small correction: with the referee's own partial-inductance primitives
the ground image alone removes 13.5 % of the free-space value (398.8 ->
345.1 pH), the lid and walls another 16.5 % (-> 279.1 pH). So the study
extends the referee to the full box by the method of images: the image of
a horizontal current segment in any PEC wall is the mirrored segment with
its current reversed (``(-1)^(a+b+c)`` for the lattice image ``(a, b, c)``;
this is the referee's own ground-image rule applied to six walls), and the
lattice sum is truncated in cubic shells ``max(|a|,|b|,|c|) <= K``. The
shell sums alternate and decay like ``1/K`` (a conditionally convergent
dipole lattice), so the partial sums are Evjen-averaged (half weight on
the outermost shell, applied twice); the raw and averaged sequences are
in the JSON (K = 4 / 6 / 8 / 10 give 279.55 / 279.13 / 279.07 / 279.04 pH).
Shell 1 uses the referee's exact bar formula (Gauss-Legendre 16^4 /
Hoer-Love, ``mutual_inductance_bars``), shell 2 its 6^4-point rule and
shells >= 3 the analytic thin-filament mutual (relative error
``O((W/d)^2) < 1e-3`` of terms that are themselves < 1 % of L). The
single-wall image sums are -53.7 (ground), -82.1 (lid), -12.7 / -15.7 (x
walls), +1.3 / -15.7 pH (y walls). The lattice is checked against the FDFD
wall by wall (``wall_check``): every wall is moved 30 um outward in both
tools on the W/2 grid and the changes of L compared. Measured: the SUM of
the three moves agrees to 0.4 % (FDFD +51.8, lattice +51.6 pH) but the
split does not (lid +20.9 vs +26.1, side walls +17.7 vs +14.9, ground
+13.1 vs +10.6 pH, i.e. ratios 0.80 / 1.19 / 1.24 on that grid); the
W/3 numbers are in the JSON. The box correction is 30 % of L, so a
per-wall disagreement of this size is a few per cent of L and is carried
as an explicit uncertainty of the referee, not hidden.

Referee-model variants (all with the box images)
------------------------------------------------
Two things separate the referee's model from a PEC strip and neither
shrinks with grid refinement. Both are quantified with the referee's own
exact bar formula and reported as variants:

* current distribution. The referee assumes a UNIFORM current density (the
  DC limit, internal inductance included); PEC metal carries its current
  on the surface, crowded toward the edges. ``shell``: every partial
  self-inductance replaced by that of a uniform surface shell (outer bar
  minus inner bar, 0.02 um thick; the value moves 1.5e-3 between 0.05
  and 0.02 um) -- removes the internal inductance, -7.2 % of L. ``pec``:
  the shell value times the ratio of the self-inductance of the
  equilibrium (skin-limit) perimeter distribution of an isolated
  rectangle to that of the uniform shell, both from an independent 2-D
  boundary-element solve of the log kernel on the perimeter
  (:func:`pec_crowding_ratio`; checks: the thin-strip GMD comes out W/4
  to 3.5e-3 at t = W/1000, the uniform-perimeter double sum reproduces
  the Hoer-Love shell to 1e-3 to 2e-3, the ratio is converged to 2e-5 in
  the perimeter mesh) -- a further -1.4 % (l = 10 um) to -1.6 % (94 um)
  per self term, -2.3 % of L (259.6 -> 253.6 pH). The
  proximity of the ground (26 um) and of the neighbouring turn (10 um)
  modifies the distribution a little more; not modelled.
* corners. Greenhouse's centreline convention has each bar end at the
  corner point: at every right-angle corner one quadrant of the W x W
  corner square is covered by both bars and the opposite quadrant by
  neither. ``corner_once``: the incoming bar is extended by W/2 to the far
  edge of the corner square and the outgoing bar shortened by W/2, so the
  corner square is covered exactly once and the total conductor length is
  unchanged (642 um); measured +1.2 % above Greenhouse's convention. The
  ``corner_none`` variant (both bars shortened by W/2) removes 8 W =
  80 um of conductor (562 um) and is reported only because an earlier
  version of this study mistook it for "corners counted once" and drew a
  14 % "corner double-count" from it; it is NOT a model of the spiral.

The surface-current referee band is therefore ``[pec, shell_corner_once]``
~ [253, 263] pH: the delivered model with the box images, minus the
internal inductance, minus edge crowding, plus/minus the corner
convention.

Gates (numbers in the JSON; the tests assert what the coarsest two levels
support)
--------
V1  Richardson-extrapolated FDFD ``L_dut`` vs the box referee within 5 %.
    The extrapolation is reported as a RANGE over four estimators (p = 1
    and p = 2 from the finest pair, the observed order from the finest
    three levels, least-squares p = 1 over all levels) because the
    observed order is not settled (0.96 from W/1-3, 1.56 from W/2-4); the
    gate passes only if every estimator is within 5 %. FAILS: measured
    L_dut 177.3 / 207.7 / 218.1 / 222.3 pH, extrapolated 228-238 pH, i.e.
    -15 to -18 % vs the delivered model with the box images, -9 to -13 %
    vs the surface-current band. The remaining gap is NOT explained by the
    referee variants above; the per-wall lattice disagreement (a few % of
    L), the FDFD's open/short residual (~1-2 %, low) and the reference
    plane (+1.8 %) are the known candidates, see the module's open issues.
V2  ``jax.grad`` vs FD4 of the same FDFD ``L_dut`` (1 % steps, which move
    L by ~1e-2 relative, far above the ~1e-11 LU noise floor measured on
    track D1) <= 1e-4 relative, all three parameters, every level. Passes
    (worst 4.2e-6, the FD4 truncation of the 1 % step on r_out).
V3  ``jax.grad`` vs FD4 of the box referee: same sign for all three
    parameters (passes at every level) and within 15 % at the finest level
    (FAILS: FDFD/referee 0.80 / 0.94 / 0.93 at W/4 for r_out / spacing /
    width; r_out sits at 0.80 from W/3 on).
V4  the per-level change of ``dL/dW`` decreases with refinement (passes:
    2.5e-6, 8.8e-7, 2.8e-7 H/m).

Scope fence. One geometry, one frequency; in-plane and vertical steps
refined together (``base_dz = 2 base_dx``, the metal slabs stay one cell
thick); the extrapolation is over ``base_dx`` (``1/N^(1/3)`` in the
figure). The referee is magnetoquasistatic with the model differences
stated above; nothing here validates losses, substrate currents or the
frequency dependence.
"""
from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
import math
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

FREQ = 1e8
N_TURNS = 2
R_OUT, SPACING, WIDTH = 52e-6, 10e-6, 10e-6
LEAD = 1.5 * WIDTH            # the column footprint [y_port, y_port + W] then sits between two mandatory lines
MARGIN = 10e-6
T_SI, T_OX_LOW, T_M1, T_VIA, T_M2, T_OX_HIGH, T_AIR = 20e-6, 1e-6, 2e-6, 2e-6, 2e-6, 3e-6, 15e-6
FD_STEP_REL = 0.01            # 1 % steps: move L by ~1e-2 relative (checked in run_level)
SHELL_DELTA = 0.02e-6         # surface-shell thickness of the "shell" referee variant (converged to 1.5e-3)
BEM_N_SIDE = 200              # perimeter segments per long side of the BEM (crowding ratio converged to 3e-4)
WALL_SHIFT = 30e-6            # wall_check: every wall moved this far outward
PARAMS = ("r_out", "spacing", "width")
VARIANTS = ("ground_only", "box", "box_shell", "box_pec", "box_corner_once", "box_shell_corner_once",
            "box_pec_corner_once", "box_corner_none")


# ----------------------------------------------------------------------------
# referee (host numpy; loaded from its file, the comparators directory has no package)

def load_referee():
    spec = importlib.util.spec_from_file_location("spiral_greenhouse", REFEREE_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class BoxFrame:
    """The FDFD box in the referee's coordinates (strip centreline at
    z = 0): ``x0, x1, y0, y1`` the side walls, ``h`` the ground depth,
    ``lid`` the lid height, ``dz_under`` the underpass centre depth."""

    def __init__(self, x0: float, x1: float, y0: float, y1: float, h: float, lid: float, dz_under: float):
        self.x0, self.x1, self.y0, self.y1, self.h, self.lid, self.dz_under = x0, x1, y0, y1, h, lid, dz_under

    @classmethod
    def from_model(cls, model) -> "BoxFrame":
        st = model.spec.stack
        zc2 = st.z_m2 + 0.5 * st.t_m2
        zc1 = st.z_m1 + 0.5 * st.t_m1
        return cls(float(model.x_nom[0]), float(model.x_nom[-1]), float(model.y_nom[0]), float(model.y_nom[-1]),
                   zc2, float(model.z[-1]) - zc2, zc2 - zc1)

    def as_dict(self) -> dict[str, float]:
        return {"x0": self.x0, "x1": self.x1, "y0": self.y0, "y1": self.y1, "ground_depth_h": self.h,
                "lid_height": self.lid, "underpass_depth": self.dz_under}

    @classmethod
    def from_dict(cls, d: dict[str, float]) -> "BoxFrame":
        return cls(d["x0"], d["x1"], d["y0"], d["y1"], d["ground_depth_h"], d["lid_height"], d["underpass_depth"])


def referee_segments(sg, theta: Sequence[float], frame: BoxFrame, lead: float = LEAD, reference_plane_shift: float | None = None):
    """Referee segment list for ``theta = (r_out, spacing, width)``: the
    lead ends at the centre of the FDFD column footprint (``lead - W/2``
    unless ``reference_plane_shift`` overrides the ``W/2``)."""
    r_out, spacing, width = (float(v) for v in theta)
    shift = 0.5 * width if reference_plane_shift is None else reference_plane_shift
    return sg.rect_spiral_segments(N_TURNS, r_out, width, spacing, T_M2, lead_length=lead - shift,
                                   underpass=(frame.dz_under, T_M1))


def conductor_length(segs) -> float:
    """Sum of the centreline lengths (m)."""
    return float(sum(np.linalg.norm(np.asarray(s.end, float) - np.asarray(s.start, float)) for s in segs))


def _image_segment(sg, seg, a: int, b: int, c: int, fr: BoxFrame):
    """Lattice image ``(a, b, c)`` of a horizontal segment: even indices
    translate by the box period, odd ones mirror; every reflection reverses
    the current (tangential to every wall the segments are horizontal, and
    the normal component of an x-directed segment in an x-wall is preserved
    together with its reversed geometric direction -- the same rule)."""
    def mp(v: float, i: int, lo: float, hi: float) -> float:
        span = hi - lo
        return v + i * span if i % 2 == 0 else 2.0 * lo - v + (i + 1) * span

    st = (mp(seg.start[0], a, fr.x0, fr.x1), mp(seg.start[1], b, fr.y0, fr.y1), mp(seg.start[2], c, -fr.h, fr.lid))
    en = (mp(seg.end[0], a, fr.x0, fr.x1), mp(seg.end[1], b, fr.y0, fr.y1), mp(seg.end[2], c, -fr.h, fr.lid))
    return sg.Segment(st, en, seg.width, seg.thickness, seg.current * (-1) ** (a + b + c))


def _pair_tiered(sg, si, sj, ngl: int) -> float:
    """Signed mutual of two segments: zero when perpendicular; the
    referee's exact ``mutual_inductance_bars`` when the cross-sections
    are closer than a cross-section dimension (collinear translation images
    included), else a ``ngl^4``-point Gauss-Legendre rule (``ngl = 16``
    is the referee's own) or, for ``ngl = 0``, the analytic thin-filament
    mutual at the centreline distance."""
    ai, aj = sg.segment_axis(si), sg.segment_axis(sj)
    if ai != aj:
        return 0.0
    li, wi, ti, cwi, cti, cli, oi = sg._bar_frame(si, ai)
    lj, wj, tj, cwj, ctj, clj, oj = sg._bar_frame(sj, aj)
    dw, dt, dl = cwj - cwi, ctj - cti, clj - cli
    if ngl == 16 or sg.cross_section_gap(wi, ti, wj, tj, dw, dt) < max(wi, ti, wj, tj):
        return oi * oj * sg.mutual_inductance_bars(li, wi, ti, lj, wj, tj, dw, dt, dl)
    if ngl == 0:
        d = math.hypot(dw + 0.5 * (wj - wi), dt + 0.5 * (tj - ti))
        return oi * oj * sg.filament_mutual(d, li, lj, dl)
    return oi * oj * sg._gauss_legendre_bars(li, wi, ti, lj, wj, tj, dw, dt, dl, ngl)


def lattice_shells(sg, segs, frame: BoxFrame, K: int, ngl_shell1: int = 16, ngl_shell2: int = 6) -> np.ndarray:
    """Image contributions per cubic shell ``1..K`` (index 0 unused):
    ``sum_{real i} sum_{image j} M_ij`` over the images with
    ``max(|a|,|b|,|c|) = shell``."""
    out = np.zeros(K + 1)
    for a in range(-K, K + 1):
        for b in range(-K, K + 1):
            for c in range(-K, K + 1):
                if (a, b, c) == (0, 0, 0):
                    continue
                shell = max(abs(a), abs(b), abs(c))
                ngl = ngl_shell1 if shell == 1 else (ngl_shell2 if shell == 2 else 0)
                imgs = [_image_segment(sg, s, a, b, c, frame) for s in segs]
                out[shell] += sum(_pair_tiered(sg, si, sj, ngl) for si in segs for sj in imgs)
    return out


def single_wall_images(sg, segs, frame: BoxFrame, ngl: int = 16) -> dict[str, float]:
    """The six single-reflection image sums (H): what each wall alone
    would do by the image method (informational; the lattice is not the
    sum of these)."""
    keys = {"x0": (-1, 0, 0), "x1": (1, 0, 0), "y0": (0, -1, 0), "y1": (0, 1, 0), "ground": (0, 0, -1), "lid": (0, 0, 1)}
    out = {}
    for name, (a, b, c) in keys.items():
        imgs = [_image_segment(sg, s, a, b, c, frame) for s in segs]
        out[name] = float(sum(_pair_tiered(sg, si, sj, ngl) for si in segs for sj in imgs))
    return out


def evjen(free: float, shells: np.ndarray) -> dict[str, list[float]]:
    """Partial sums ``S_K`` and the Evjen averages ``A1 = (S_{K-1} +
    S_K)/2``, ``A2 = (A1_{K-1} + A1_K)/2``; the last ``A2`` is the value."""
    s = free + np.cumsum(shells)
    a1 = 0.5 * (s[1:] + s[:-1])
    a2 = 0.5 * (a1[1:] + a1[:-1])
    value = float(a2[-1] if len(a2) else (a1[-1] if len(a1) else s[-1]))
    return {"partial_sums": s.tolist(), "evjen1": a1.tolist(), "evjen2": a2.tolist(), "value": value}


def box_value(sg, segs, frame: BoxFrame, K: int, ngl_shell1: int, ngl_shell2: int) -> tuple[Any, np.ndarray, dict]:
    """``(greenhouse_terms, shells, evjen)`` of a segment list in the box."""
    g = sg.greenhouse_terms(segs)
    shells = lattice_shells(sg, segs, frame, K, ngl_shell1, ngl_shell2)
    return g, shells, evjen(g.total, shells)


# ----------------------------------------------------------------------------
# current-distribution variants of the self terms

def shell_self_inductance(sg, length: float, w: float, t: float, delta: float) -> float:
    """Partial self-inductance of a bar whose current flows in a uniform
    surface shell of thickness ``delta``: the shell is the outer bar minus
    the inner bar, so ``L = (Ao^2 Loo - 2 Ao Ai Moi + Ai^2 Lii) / (Ao -
    Ai)^2`` with the referee's exact Hoer-Love terms."""
    wi, ti = w - 2 * delta, t - 2 * delta
    ao, ai = w * t, wi * ti
    loo = sg.mutual_inductance_bars(length, w, t, length, w, t, 0.0, 0.0, 0.0, method="hoer_love")
    lii = sg.mutual_inductance_bars(length, wi, ti, length, wi, ti, 0.0, 0.0, 0.0, method="hoer_love")
    moi = sg.mutual_inductance_bars(length, w, t, length, wi, ti, delta, delta, 0.0, method="hoer_love")
    return (ao ** 2 * loo - 2 * ao * ai * moi + ai ** 2 * lii) / (ao - ai) ** 2


def perimeter_mesh(w: float, t: float, n_side: int) -> tuple[np.ndarray, np.ndarray]:
    """Boundary elements on the perimeter of a ``w x t`` rectangle:
    midpoints ``(n, 2)`` and lengths ``(n,)``; ``n_side`` elements on each
    long side (proportionally fewer on the short ones, at least 4),
    cosine-clustered toward the corners where the equilibrium density
    diverges like ``r^(-1/3)``."""
    corners = [(-w / 2, -t / 2), (w / 2, -t / 2), (w / 2, t / 2), (-w / 2, t / 2)]
    p0, p1 = [], []
    for k in range(4):
        a, b = np.asarray(corners[k]), np.asarray(corners[(k + 1) % 4])
        n = n_side if k % 2 == 0 else max(4, int(round(n_side * t / w)))
        s = 0.5 - 0.5 * np.cos(np.pi * np.linspace(0.0, 1.0, n + 1))
        for i in range(n):
            p0.append(a + s[i] * (b - a))
            p1.append(a + s[i + 1] * (b - a))
    a0, a1 = np.asarray(p0), np.asarray(p1)
    return 0.5 * (a0 + a1), np.linalg.norm(a1 - a0, axis=1)


def equilibrium_distribution(mid: np.ndarray, length: np.ndarray) -> np.ndarray:
    """Charge per element (sums to 1) that makes the 2-D log potential
    constant on the contour: the electrostatic equilibrium of the
    cross-section, which is also the skin-effect-limit (PEC) current
    distribution of an isolated straight conductor (both minimise the
    same 2-D log energy). Collocation at the midpoints; the self term is
    the midpoint potential of a uniform element, ``-(ln(a/2) - 1)``."""
    n = len(length)
    d = np.linalg.norm(mid[:, None, :] - mid[None, :, :], axis=2)
    k = -np.log(np.where(d > 0, d, 1.0))
    k[np.arange(n), np.arange(n)] = -(np.log(0.5 * length) - 1.0)
    q = np.linalg.solve(k, np.ones(n))
    return q / q.sum()


def bem_self_inductance(sg, length: float, mid: np.ndarray, elem: np.ndarray, q: np.ndarray) -> float:
    """Partial self-inductance of a bar of ``length`` whose current is
    distributed over the perimeter elements with weights ``q``: the double
    sum of the referee's thin-filament mutual over element pairs, the
    self pairs at the element's own geometric mean distance
    ``a e^(-3/2)``."""
    d = np.linalg.norm(mid[:, None, :] - mid[None, :, :], axis=2)
    n = len(elem)
    d[np.arange(n), np.arange(n)] = elem * math.exp(-1.5)
    m = sg._filament_mutual_array(d, length, length, 0.0)
    return float(q @ m @ q)


def log_gmd(mid: np.ndarray, elem: np.ndarray, q: np.ndarray) -> float:
    """``ln`` of the geometric mean distance of the distribution ``q``
    from itself (self pairs at ``a e^(-3/2)``)."""
    d = np.linalg.norm(mid[:, None, :] - mid[None, :, :], axis=2)
    lnd = np.log(np.where(d > 0, d, 1.0))
    n = len(elem)
    lnd[np.arange(n), np.arange(n)] = np.log(elem) - 1.5
    return float(q @ lnd @ q)


_BEM_CACHE: dict[tuple[float, float, int], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}


def pec_crowding_ratio(sg, length: float, w: float, t: float, n_side: int = BEM_N_SIDE) -> float:
    """``L_equilibrium / L_uniform_perimeter`` for a bar of ``length`` and
    section ``w x t`` (both from :func:`bem_self_inductance`): the factor
    by which edge crowding lowers the self-inductance of a surface-current
    bar below the uniform shell. Measured for 10 x 2 um: -1.68 % at
    l = 94 um, -1.44 % at 10 um, converged to 2e-5 for n_side >= 200."""
    key = (w, t, n_side)
    if key not in _BEM_CACHE:
        mid, elem = perimeter_mesh(w, t, n_side)
        _BEM_CACHE[key] = (mid, elem, elem / elem.sum(), equilibrium_distribution(mid, elem))
    mid, elem, qu, qe = _BEM_CACHE[key]
    return bem_self_inductance(sg, length, mid, elem, qe) / bem_self_inductance(sg, length, mid, elem, qu)


def variant_self_sum(sg, segs, kind: str, delta: float = SHELL_DELTA) -> float:
    """Sum over the segments of the self term of the ``kind``: ``"shell"``
    (:func:`shell_self_inductance`) or ``"pec"`` (the shell value times
    :func:`pec_crowding_ratio`)."""
    total = 0.0
    for s in segs:
        length, w, t = sg._bar_frame(s, sg.segment_axis(s))[:3]
        v = shell_self_inductance(sg, length, w, t, delta)
        if kind == "pec":
            v *= pec_crowding_ratio(sg, length, w, t)
        elif kind != "shell":
            raise ValueError(kind)
        total += v
    return total


# ----------------------------------------------------------------------------
# corner conventions

def _is_corner(segs, k: int, j: int) -> bool:
    """Segments ``k`` and ``j`` (adjacent in the chain) meet at a
    right-angle corner on the same level."""
    if j < 0 or j >= len(segs):
        return False
    dk = np.asarray(segs[k].end, float) - np.asarray(segs[k].start, float)
    dj = np.asarray(segs[j].end, float) - np.asarray(segs[j].start, float)
    return bool(abs(float(np.dot(dk, dj))) < 0.5 * np.linalg.norm(dk) * np.linalg.norm(dj)
                and segs[k].start[2] == segs[j].start[2])


def _reshape_corners(sg, segs, incoming: float, outgoing: float):
    """Move every corner end: the incoming bar's end by ``incoming * W/2``
    along its direction, the outgoing bar's start by ``outgoing * W/2``."""
    out = []
    for k, s in enumerate(segs):
        a, b = np.asarray(s.start, float), np.asarray(s.end, float)
        d = (b - a) / np.linalg.norm(b - a)
        if _is_corner(segs, k, k - 1):
            a = a + d * outgoing * 0.5 * s.width
        if _is_corner(segs, k, k + 1):
            b = b + d * incoming * 0.5 * s.width
        out.append(sg.Segment(tuple(a), tuple(b), s.width, s.thickness, s.current))
    return out


def corner_once_segments(sg, segs):
    """Each corner square covered exactly once: the incoming bar extended
    by ``W/2`` to the far edge of the corner square, the outgoing bar
    started ``W/2`` later. Total length unchanged; collinear joints (lead /
    first side, via) untouched."""
    return _reshape_corners(sg, segs, incoming=+1.0, outgoing=+1.0)


def corner_none_segments(sg, segs):
    """Both bars shortened by ``W/2`` at every corner: the corner squares
    are covered by NEITHER bar and ``8 W`` of conductor is missing. Not a
    model of the spiral; reported as the lower end of the corner-convention
    spread only (see the module docstring)."""
    return _reshape_corners(sg, segs, incoming=-1.0, outgoing=+1.0)


# ----------------------------------------------------------------------------
# the referee value and its variants

def referee_L(sg, theta: Sequence[float], frame: BoxFrame, K: int = 8, ngl_shell1: int = 16, ngl_shell2: int = 6,
              variants: bool | Sequence[str] = True, lead: float = LEAD) -> dict[str, Any]:
    """Referee inductances (H) for ``theta``: ``free`` (no walls),
    ``ground_only`` (the delivered referee, one image), ``box`` (the full
    image lattice, Evjen-averaged) for the delivered uniform-current
    centreline model, plus -- with ``variants`` -- the ``shell`` / ``pec``
    current-distribution variants and the ``corner_once`` /
    ``corner_none`` conventions (``box_<current>_<corner>``), the
    conductor lengths and the surface-current band
    ``[box_pec, box_shell_corner_once]``. ``variants`` may also be the
    names of the variants wanted (only their lattices are evaluated)."""
    segs = referee_segments(sg, theta, frame, lead=lead)
    g, shells, ev = box_value(sg, segs, frame, K, ngl_shell1, ngl_shell2)
    box_corr = ev["value"] - g.total
    out: dict[str, Any] = {
        "free": g.total, "self_sum": g.self_sum, "mutual_sum": g.mutual_sum,
        "ground_only": sg.greenhouse_inductance(segs, ground_height=frame.h),
        "box": ev["value"], "box_correction": box_corr, "shells": shells[1:].tolist(),
        "lattice": ev, "K": K, "worst_error": g.worst_error, "conductor_length": conductor_length(segs),
    }
    wanted = set(VARIANTS[2:]) if variants is True else (set() if variants is False else set(variants))
    if wanted & {"box_shell", "box_pec"}:
        out["box_shell"] = g.total - g.self_sum + variant_self_sum(sg, segs, "shell") + box_corr
        out["box_pec"] = g.total - g.self_sum + variant_self_sum(sg, segs, "pec") + box_corr
    if wanted & {"box_corner_once", "box_shell_corner_once", "box_pec_corner_once"}:
        segs_once = corner_once_segments(sg, segs)
        g1, _, ev1 = box_value(sg, segs_once, frame, K, ngl_shell1, ngl_shell2)
        corr1 = ev1["value"] - g1.total
        out["box_corner_once"] = ev1["value"]
        out["box_shell_corner_once"] = g1.total - g1.self_sum + variant_self_sum(sg, segs_once, "shell") + corr1
        out["box_pec_corner_once"] = g1.total - g1.self_sum + variant_self_sum(sg, segs_once, "pec") + corr1
        out["conductor_length_corner_once"] = conductor_length(segs_once)
    if "box_corner_none" in wanted:
        segs_none = corner_none_segments(sg, segs)
        g0, _, ev0 = box_value(sg, segs_none, frame, K, ngl_shell1, ngl_shell2)
        out["box_corner_none"] = ev0["value"]
        out["conductor_length_corner_none"] = conductor_length(segs_none)
    if "box_pec" in out and "box_shell_corner_once" in out:
        out["surface_current_band"] = [out["box_pec"], out["box_shell_corner_once"]]
    return out


def fd4(f: Callable[[float], float], x0: float, d: float) -> float:
    return (-f(x0 + 2 * d) + 8 * f(x0 + d) - 8 * f(x0 - d) + f(x0 - 2 * d)) / (12 * d)


def referee_gradient(sg, theta: Sequence[float], frame: BoxFrame, key: str = "box", K: int = 8,
                     ngl_shell1: int = 16, ngl_shell2: int = 6, step_rel: float = FD_STEP_REL) -> np.ndarray:
    """FD4 of ``referee_L(...)[key]`` with relative steps ``step_rel``
    (only the variant ``key`` needs is evaluated)."""
    th = np.asarray(theta, float)
    g = np.zeros(3)
    need: bool | Sequence[str] = (key,) if key in VARIANTS[2:] else False
    for k in range(3):
        def f(t: float, k: int = k) -> float:
            th2 = th.copy()
            th2[k] = t
            return referee_L(sg, th2, frame, K, ngl_shell1, ngl_shell2, variants=need)[key]
        g[k] = fd4(f, th[k], step_rel * th[k])
    return g


# ----------------------------------------------------------------------------
# FDFD levels

def study_spec(div: int):
    """``SpiralSpec`` of level ``div``: ``base_dx = W / div``, ``base_dz =
    2 base_dx`` (silicon and air get ``div`` cells each, every metal slab
    one)."""
    from rfx.fdfd import spiral as sm
    dx = WIDTH / div
    stack = sm.SmallStack(t_si=T_SI, t_ox_low=T_OX_LOW, t_m1=T_M1, t_via=T_VIA, t_m2=T_M2, t_ox_high=T_OX_HIGH,
                          t_air=T_AIR, base_dz=2 * dx, metal_cells=1)
    return sm.SpiralSpec(n_turns=N_TURNS, r_out=R_OUT, spacing=SPACING, width=WIDTH, lead=LEAD, base_dx=dx,
                         margin=MARGIN, stack=stack)


def run_level(div: int, log: Callable[[str], None] = print, fd4_params: Sequence[int] = (0, 1, 2)) -> dict[str, Any]:
    """Build and solve level ``div``: de-embedded and raw L, ``jax.grad``
    and FD4 of ``L_dut`` (1 % steps) for the parameters in ``fd4_params``
    (``nan`` for the others), sizes and times
    (``solve_seconds_three_fixtures`` is the mean of the jitted FD4
    solves, each a fresh three-fixture factorisation)."""
    import jax
    import jax.numpy as jnp
    from rfx.fdfd import spiral as sm
    from rfx.fdfd.linear_solve import clear_factor_cache

    if not jax.config.read("jax_enable_x64"):
        raise RuntimeError("run_level needs x64")
    model = sm.build_spiral(study_spec(div))
    th0 = jnp.asarray(model.theta_nominal)

    def l_dut(theta):
        return sm.solve_spiral(model, FREQ, theta).L_diff

    def l_dut_aux(theta):
        r = sm.solve_spiral(model, FREQ, theta)
        return r.L_diff, (r.L_raw, r.L_se)

    clear_factor_cache()
    t0 = time.time()
    (v, (l_raw, l_se)), g = jax.value_and_grad(l_dut_aux, has_aux=True)(th0)
    jax.block_until_ready(g)
    t_grad = time.time() - t0
    v, g = float(v), np.asarray(g, float)
    l_jit = jax.jit(l_dut)
    t0 = time.time()
    v_jit = float(l_jit(th0))
    t_jit_first = time.time() - t0
    fds, moves, t_fd, n_solves = np.full(3, np.nan), np.full(3, np.nan), 0.0, 0
    for k in fd4_params:
        e = np.zeros(3)
        e[k] = 1.0
        d = FD_STEP_REL * float(th0[k])

        def f(t: float) -> float:
            clear_factor_cache()
            return float(l_jit(th0 + t * e))

        t0 = time.time()
        fds[k] = fd4(f, 0.0, d)
        t_fd += time.time() - t0
        n_solves += 4
        moves[k] = abs(fds[k] * d) / v
    rel = np.abs(g - fds) / np.abs(fds)
    rec = {
        "div": div, "base_dx": WIDTH / div, "base_dz": 2 * WIDTH / div, "shape": list(model.shape),
        "n_unknowns": model.n_unknowns, "build_seconds": model.build_seconds,
        "solve_seconds_three_fixtures": t_fd / max(n_solves, 1), "jit_compile_and_first_solve_seconds": t_jit_first,
        "value_and_grad_seconds": t_grad, "fd4_seconds": t_fd, "fd4_solves": n_solves,
        "L_dut": v, "L_raw": float(l_raw), "L_se_y11": float(l_se), "jit_minus_eager_rel": abs(v_jit - v) / v,
        "grad": g.tolist(), "fd4": fds.tolist(), "fd4_step_moves_L_rel": moves.tolist(),
        "grad_vs_fd4_rel": rel.tolist(), "frame": BoxFrame.from_model(model).as_dict(),
    }
    log(f"level W/{div}: shape {model.shape} N {model.n_unknowns} solve {rec['solve_seconds_three_fixtures']:.1f} s "
        f"grad {t_grad:.1f} s FD4 {t_fd:.1f} s  L_dut {v * 1e12:.3f} pH  grad {g}  grad/FD4-1 {rel}")
    return rec


def wall_check(sg, div: int, K: int = 8, ngl_shell1: int = 16, ngl_shell2: int = 6, shift: float = WALL_SHIFT,
               log: Callable[[str], None] = print) -> dict[str, Any]:
    """Lattice vs FDFD wall by wall on the level-``div`` grid: the lid, the
    four side walls and the ground are each moved ``shift`` outward (more
    air / margin / silicon, same ``base_dx``) and the change of ``L_dut``
    in the FDFD is compared with the change of the box-lattice referee
    for the same frames. The FDFD's discretisation error of the spiral
    itself largely cancels in the difference; the referee's model
    differences (current distribution, corners) cancel exactly."""
    import jax
    from rfx.fdfd import spiral as sm
    from rfx.fdfd.linear_solve import clear_factor_cache

    if not jax.config.read("jax_enable_x64"):
        raise RuntimeError("wall_check needs x64")
    theta = (R_OUT, SPACING, WIDTH)
    base = study_spec(div)
    specs = {
        "nominal": base,
        "lid": dataclasses.replace(base, stack=dataclasses.replace(base.stack, t_air=base.stack.t_air + shift)),
        "side_walls": dataclasses.replace(base, margin=base.margin + shift),
        "ground": dataclasses.replace(base, stack=dataclasses.replace(base.stack, t_si=base.stack.t_si + shift)),
    }
    rows: dict[str, Any] = {}
    for name, sp in specs.items():
        model = sm.build_spiral(sp)
        clear_factor_cache()
        t0 = time.time()
        l_fdfd = float(sm.solve_spiral(model, FREQ).L_diff)
        t_solve = time.time() - t0
        frame = BoxFrame.from_model(model)
        segs = referee_segments(sg, theta, frame)
        l_ref = box_value(sg, segs, frame, K, ngl_shell1, ngl_shell2)[2]["value"]
        rows[name] = {"L_fdfd": l_fdfd, "L_lattice": l_ref, "n_unknowns": model.n_unknowns,
                      "solve_seconds": t_solve, "frame": frame.as_dict()}
        log(f"wall_check W/{div} {name}: N {model.n_unknowns} FDFD {l_fdfd * 1e12:.3f} lattice {l_ref * 1e12:.3f} pH")
    out: dict[str, Any] = {"div": div, "shift": shift, "rows": rows, "delta_fdfd": {}, "delta_lattice": {}, "ratio": {}}
    for name in ("lid", "side_walls", "ground"):
        df = rows[name]["L_fdfd"] - rows["nominal"]["L_fdfd"]
        dl = rows[name]["L_lattice"] - rows["nominal"]["L_lattice"]
        out["delta_fdfd"][name], out["delta_lattice"][name], out["ratio"][name] = df, dl, df / dl
    out["sum_delta_fdfd"] = sum(out["delta_fdfd"].values())
    out["sum_delta_lattice"] = sum(out["delta_lattice"].values())
    out["sum_ratio"] = out["sum_delta_fdfd"] / out["sum_delta_lattice"]
    log(f"wall_check W/{div}: ratios FDFD/lattice {out['ratio']} sum {out['sum_ratio']:.3f}")
    return out


# ----------------------------------------------------------------------------
# extrapolation

def richardson(hs: Sequence[float], ls: Sequence[float]) -> dict[str, Any]:
    """Least-squares ``L* + C h^p`` for ``p = 1`` and ``p = 2`` over all
    levels (residual reported), the two-level ``p = 1`` and ``p = 2``
    values from the finest pair, and the observed order from the finest
    three levels (root of ``(L1 - L2) / (L2 - L3) = (h1^p - h2^p) / (h2^p -
    h3^p)``) with its extrapolation. ``estimates`` collects the
    extrapolated values by name and ``range`` their min / max."""
    h = np.asarray(hs, float)
    lv = np.asarray(ls, float)
    out: dict[str, Any] = {}
    est: dict[str, float] = {}
    for p in (1, 2):
        a = np.stack([np.ones_like(h), h ** p], axis=1)
        coef, *_ = np.linalg.lstsq(a, lv, rcond=None)
        resid = lv - a @ coef
        out[f"p{p}"] = {"L_extrapolated": float(coef[0]), "C": float(coef[1]),
                        "rms_residual_rel": float(np.sqrt(np.mean(resid ** 2)) / abs(coef[0]))}
    if len(h) >= 2:
        h1, h2 = h[-2], h[-1]
        out["two_level_p1"] = float((lv[-1] * h1 - lv[-2] * h2) / (h1 - h2))
        out["two_level_p2"] = float((lv[-1] * h1 ** 2 - lv[-2] * h2 ** 2) / (h1 ** 2 - h2 ** 2))
        est["p1_finest_pair"] = out["two_level_p1"]
        est["p2_finest_pair"] = out["two_level_p2"]
    if len(h) >= 3:
        est["p1_lstsq_all"] = out["p1"]["L_extrapolated"]
        h1, h2, h3 = h[-3:]
        l1, l2, l3 = lv[-3:]
        target = (l1 - l2) / (l2 - l3)

        def resid_p(p: float) -> float:
            return (h1 ** p - h2 ** p) / (h2 ** p - h3 ** p) - target

        ps = np.linspace(0.2, 4.0, 3801)
        vals = np.array([resid_p(p) for p in ps])
        sign = np.sign(vals)
        idx = np.nonzero(sign[:-1] * sign[1:] < 0)[0]
        if len(idx):
            lo, hi = ps[idx[0]], ps[idx[0] + 1]
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                if resid_p(lo) * resid_p(mid) <= 0:
                    hi = mid
                else:
                    lo = mid
            p_obs = 0.5 * (lo + hi)
            out["observed_order"] = float(p_obs)
            out["L_extrapolated_observed_order"] = float(l3 + (l3 - l2) * h3 ** p_obs / (h2 ** p_obs - h3 ** p_obs))
            est["observed_order_finest_three"] = out["L_extrapolated_observed_order"]
        else:
            out["observed_order"] = None
            out["L_extrapolated_observed_order"] = None
    out["estimates"] = est
    out["range"] = [min(est.values()), max(est.values())] if est else [float(lv[-1]), float(lv[-1])]
    return out


# ----------------------------------------------------------------------------
# study

def run_study(levels: Sequence[int], K: int = 8, log: Callable[[str], None] = print,
              existing: dict[str, Any] | None = None, ngl_shell1: int = 16, ngl_shell2: int = 6,
              shell_gradient: bool = True, fd4_params_per_level: dict[int, Sequence[int]] | None = None,
              wall_check_level: int | None = 2) -> dict[str, Any]:
    """The whole study: FDFD levels, referee and its variants, referee
    FD4 gradients, the wall check, Richardson fits and the gate numbers.
    ``K`` / ``ngl_shell1`` / ``ngl_shell2`` set the image-lattice tier (the
    test uses ``K = 4, 8, 4``: measured 1.7e-3 from the ``K = 8, 16, 6``
    default); ``shell_gradient`` adds the FD4 gradients of the ``box_shell``
    and ``box_pec_corner_once`` variants; ``fd4_params_per_level`` restricts
    the FD4 parameters of a level (the test's W/2 runs r_out only);
    ``wall_check_level`` runs :func:`wall_check` on that level (``None``
    skips it). ``existing["levels"]`` records are kept for levels not
    re-run."""
    import jax
    if not jax.config.read("jax_enable_x64"):
        raise RuntimeError("run_study needs x64 (main() enables it; tests scope it)")
    sg = load_referee()
    per_level: dict[str, Any] = dict((existing or {}).get("levels", {}))
    for div in levels:
        params = (fd4_params_per_level or {}).get(div, (0, 1, 2))
        per_level[str(div)] = run_level(div, log, fd4_params=params)
    divs = sorted(int(k) for k in per_level)
    frame = BoxFrame.from_dict(per_level[str(divs[0])]["frame"])
    theta = (R_OUT, SPACING, WIDTH)
    t0 = time.time()
    ref = referee_L(sg, theta, frame, K, ngl_shell1, ngl_shell2)
    t_ref = time.time() - t0
    ref_full_lead = referee_L(sg, theta, frame, K, ngl_shell1, ngl_shell2, variants=False,
                              lead=LEAD + 0.5 * WIDTH)  # lead ends at y_port instead of the column centre
    walls = single_wall_images(sg, referee_segments(sg, theta, frame), frame)
    t0 = time.time()
    g_ref = referee_gradient(sg, theta, frame, "box", K, ngl_shell1, ngl_shell2)
    g_ref_ground = referee_gradient(sg, theta, frame, "ground_only", K=1, ngl_shell1=0)
    t_gref = time.time() - t0
    g_ref_var = {"box_shell": np.full(3, np.nan), "box_pec_corner_once": np.full(3, np.nan)}
    if shell_gradient:
        for key in g_ref_var:
            g_ref_var[key] = referee_gradient(sg, theta, frame, key, K, ngl_shell1, ngl_shell2)
    log(f"referee ({t_ref:.1f} s): free {ref['free'] * 1e12:.2f} ground-only {ref['ground_only'] * 1e12:.2f} "
        f"box {ref['box'] * 1e12:.2f} shell {ref['box_shell'] * 1e12:.2f} pec {ref['box_pec'] * 1e12:.2f} "
        f"corner-once {ref['box_corner_once'] * 1e12:.2f} shell+once {ref['box_shell_corner_once'] * 1e12:.2f} "
        f"pec+once {ref['box_pec_corner_once'] * 1e12:.2f} corner-none {ref['box_corner_none'] * 1e12:.2f} pH; "
        f"grad box {g_ref} ({t_gref:.1f} s)")
    wall = wall_check(sg, wall_check_level, K, ngl_shell1, ngl_shell2, log=log) if wall_check_level else None

    hs = [per_level[str(d)]["base_dx"] for d in divs]
    ls = [per_level[str(d)]["L_dut"] for d in divs]
    rich_all = richardson(hs, ls)
    rich_fine = richardson(hs[-3:], ls[-3:]) if len(divs) >= 3 else None
    estimates = dict(rich_all["estimates"])
    l_range = rich_all["range"]
    gaps = {name: {str(d): per_level[str(d)]["L_dut"] / ref[name] - 1.0 for d in divs} for name in VARIANTS}
    ext_gap = {name: {est: v / ref[name] - 1.0 for est, v in estimates.items()} for name in VARIANTS}
    gap_range = {name: [l_range[0] / ref[name] - 1.0, l_range[1] / ref[name] - 1.0] for name in VARIANTS}
    band = ref["surface_current_band"]
    band_gap = [l_range[0] / band[0] - 1.0, l_range[1] / band[1] - 1.0]   # < 0 / > 0 when outside on that side
    gap_seq = [abs(gaps["box"][str(d)]) for d in divs]
    grads = np.array([per_level[str(d)]["grad"] for d in divs])
    fds = np.array([per_level[str(d)]["fd4"] for d in divs])
    v2 = np.abs(grads - fds) / np.abs(fds)                       # nan where a parameter was not FD-checked
    v2_worst = float(np.nanmax(v2))
    ratio = grads / g_ref
    ratio_var = {key: grads / g for key, g in g_ref_var.items()}
    dw_changes = np.abs(np.diff(grads[:, 2])) if len(divs) >= 2 else np.zeros(0)
    gates = {
        "V1": {"gate": "|L_extrapolated / L_ref - 1| <= 0.05 for EVERY extrapolation estimator (vs 'box')",
               "L_extrapolated_estimates": estimates, "L_extrapolated_range": l_range,
               "gap_extrapolated": ext_gap, "gap_range": gap_range, "gap_per_level": gaps,
               "gap_shrinks_with_refinement_box": bool(all(np.diff(gap_seq) < 0)) if len(divs) >= 2 else None,
               "passed": bool(max(abs(v) for v in gap_range["box"]) <= 0.05),
               "passed_per_variant": {name: bool(max(abs(v) for v in gap_range[name]) <= 0.05) for name in VARIANTS},
               "band_gate": "L_extrapolated range inside the surface-current band [box_pec, box_shell_corner_once]",
               "surface_current_band": band, "band_gap": band_gap,
               "passed_surface_current_band": bool(band[0] <= l_range[0] and l_range[1] <= band[1])},
        "V2": {"gate": "|grad / FD4 - 1| <= 1e-4, every FD-checked parameter, every level",
               "per_level": {str(d): v2[i].tolist() for i, d in enumerate(divs)},
               "worst": v2_worst, "passed": bool(v2_worst <= 1e-4)},
        "V3": {"gate": "same sign all parameters; |grad / grad_ref - 1| <= 0.15 at the finest level",
               "ratio_per_level": {str(d): ratio[i].tolist() for i, d in enumerate(divs)},
               "ratio_per_level_vs_variant": {key: {str(d): r[i].tolist() for i, d in enumerate(divs)}
                                              for key, r in ratio_var.items()},
               "same_sign_all_levels": bool(np.all(np.sign(grads) == np.sign(g_ref))),
               "finest_worst": float(np.abs(ratio[-1] - 1.0).max()),
               "passed": bool(np.all(np.sign(grads[-1]) == np.sign(g_ref)) and np.abs(ratio[-1] - 1.0).max() <= 0.15),
               "finest_worst_vs_variant": {key: float(np.abs(r[-1] - 1.0).max()) for key, r in ratio_var.items()}},
        "V4": {"gate": "|dL/dW(level k+1) - dL/dW(level k)| decreasing",
               "dL_dW_per_level": grads[:, 2].tolist(), "changes": dw_changes.tolist(),
               "passed": bool(all(np.diff(dw_changes) < 0)) if len(dw_changes) >= 2 else None},
    }
    return {
        "geometry": {"n_turns": N_TURNS, "r_out": R_OUT, "spacing": SPACING, "width": WIDTH, "lead": LEAD,
                     "margin": MARGIN, "stack": {"t_si": T_SI, "t_ox_low": T_OX_LOW, "t_m1": T_M1, "t_via": T_VIA,
                                                 "t_m2": T_M2, "t_ox_high": T_OX_HIGH, "t_air": T_AIR},
                     "freq": FREQ, "referee_frame": frame.as_dict(),
                     "referee_lead_length": LEAD - 0.5 * WIDTH},
        "levels": per_level,
        "referee": {**ref, "gradient_box_fd4": g_ref.tolist(), "gradient_ground_only_fd4": g_ref_ground.tolist(),
                    "gradient_variants_fd4": {key: g.tolist() for key, g in g_ref_var.items()},
                    "single_wall_images": walls,
                    "reference_plane_sensitivity": {"lead_to_y_port_box": ref_full_lead["box"],
                                                    "rel_change": ref_full_lead["box"] / ref["box"] - 1.0},
                    "seconds": t_ref, "gradient_seconds": t_gref, "shell_delta": SHELL_DELTA,
                    "bem_n_side": BEM_N_SIDE},
        "wall_check": wall,
        "richardson_all_levels": rich_all, "richardson_finest_three": rich_fine,
        "gates": gates,
    }


def make_figure(study: dict[str, Any], path: pathlib.Path = PNG_PATH) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    divs = sorted(int(k) for k in study["levels"])
    lv = [study["levels"][str(d)] for d in divs]
    x = np.array([lv_["n_unknowns"] ** (-1 / 3) for lv_ in lv])
    ldut = np.array([lv_["L_dut"] for lv_ in lv]) * 1e12
    ref = study["referee"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.6))
    ax1.plot(x, ldut, "o-", color="#1f77b4", label="FDFD L_dut (de-embedded)")
    ax1.plot(x, [lv_["L_raw"] * 1e12 for lv_ in lv], "s--", color="#9ecae1", label="FDFD L_raw")
    est = study["gates"]["V1"]["L_extrapolated_estimates"]
    markers = {"p1_finest_pair": "*", "p2_finest_pair": "P", "observed_order_finest_three": "D", "p1_lstsq_all": "v"}
    for i, (name, val) in enumerate(est.items()):
        ax1.plot([0.0], [val * 1e12], marker=markers.get(name, "o"), color="#1f77b4", markersize=8, linestyle="none",
                 label=f"extrapolated {name} {val * 1e12:.1f} pH")
    lo, hi = study["gates"]["V1"]["L_extrapolated_range"]
    ax1.fill_between([0.0, x.max() * 1.1], lo * 1e12, hi * 1e12, color="#1f77b4", alpha=0.12, label="extrapolation range")
    band = ref["surface_current_band"]
    ax1.fill_between([0.0, x.max() * 1.1], band[0] * 1e12, band[1] * 1e12, color="#ff7f0e", alpha=0.18,
                     label="referee surface-current band [pec, shell+corner once]")
    for key, color, label in (("ground_only", "#d62728", "referee, ground only (delivered)"),
                              ("box", "#2ca02c", "referee + box images (uniform current, centreline)"),
                              ("box_shell", "#ff7f0e", "referee + box, surface shell"),
                              ("box_pec", "#8c564b", "referee + box, PEC edge-crowded")):
        ax1.axhline(ref[key] * 1e12, color=color, linestyle="--", linewidth=1, label=label)
    ax1.set_xlabel("N^(-1/3)")
    ax1.set_ylabel("L [pH]")
    ax1.set_xlim(0, x.max() * 1.1)
    ax1.set_title("de-embedded L vs refinement (100 MHz, PEC)")
    ax1.legend(fontsize=6, loc="lower right")
    ax1.grid(alpha=0.3)
    labels = [f"W/{d}" for d in divs]
    g = np.array([lv_["grad"] for lv_ in lv]) * 1e6
    fd = np.array([lv_["fd4"] for lv_ in lv]) * 1e6
    gref = np.array(ref["gradient_box_fd4"]) * 1e6
    gvar = np.array(ref["gradient_variants_fd4"]["box_pec_corner_once"]) * 1e6
    colors = ("#1f77b4", "#2ca02c", "#d62728")
    for k, (name, c) in enumerate(zip(PARAMS, colors)):
        ax2.plot(labels, g[:, k], "o-", color=c, label=f"dL/d{name} jax.grad")
        ax2.plot(labels, fd[:, k], "x", color=c, markersize=9, label=f"dL/d{name} FD4 (FDFD)")
        ax2.axhline(gref[k], color=c, linestyle="--", linewidth=1, label=f"dL/d{name} referee+box FD")
        ax2.axhline(gvar[k], color=c, linestyle=":", linewidth=1, label=f"dL/d{name} referee pec+corner once FD")
    ax2.set_ylabel("dL/dtheta [pH/um = uH/m]")
    ax2.set_title("shape derivatives per level")
    ax2.legend(fontsize=6, ncol=2)
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--levels", type=int, nargs="*", default=[1, 2, 3, 4], help="base_dx = W / level (may be empty with --levels-from)")
    ap.add_argument("--K", type=int, default=8, help="image-lattice shell cutoff")
    ap.add_argument("--merge", action="store_true", help="keep levels already in the JSON that are not re-run")
    ap.add_argument("--levels-from", type=pathlib.Path, default=None,
                    help="JSON with a 'levels' record (e.g. an earlier run) whose levels are kept if not re-run")
    ap.add_argument("--wall-check-level", type=int, default=2, help="0 skips the wall check")
    ap.add_argument("--json", type=pathlib.Path, default=JSON_PATH)
    ap.add_argument("--png", type=pathlib.Path, default=PNG_PATH)
    args = ap.parse_args(argv)
    existing = None
    if args.levels_from is not None:
        existing = json.loads(args.levels_from.read_text())
    elif args.merge and args.json.exists():
        existing = json.loads(args.json.read_text())
    import jax
    jax.config.update("jax_enable_x64", True)
    t0 = time.time()
    study = run_study(args.levels, args.K, existing=existing, wall_check_level=args.wall_check_level or None)
    study["total_seconds"] = time.time() - t0
    args.json.write_text(json.dumps(study, indent=2))
    make_figure(study, args.png)
    g = study["gates"]
    print(json.dumps({"V1": {k: v for k, v in g["V1"].items()
                             if k.startswith("passed") or k in ("gap_range", "L_extrapolated_range", "band_gap")},
                      "V2": {"worst": g["V2"]["worst"], "passed": g["V2"]["passed"]},
                      "V3": {"finest_worst": g["V3"]["finest_worst"], "ratio_finest": g["V3"]["ratio_per_level"][str(max(int(k) for k in study["levels"]))],
                             "passed": g["V3"]["passed"]},
                      "V4": g["V4"]}, indent=2))
    print(f"wrote {args.json} and {args.png} in {study['total_seconds']:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
