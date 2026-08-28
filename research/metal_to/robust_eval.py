"""Etch-tolerance robust evaluation for the Phase-2 dual-band notch benchmark.

Companion to the FROZEN metric in ``score_dualband.py``. That file is
pre-registered and is imported here UNCHANGED -- nothing in this module
redefines a threshold, a band edge, or an aggregation rule. All this module
does is decide *which geometries get scored* and *how the three scores are
combined into one margin number*.

Why this file exists
--------------------
PI decision, 2026-08-27: the phase is a MARGIN COMPARISON, not a feasibility
gate, and tolerance robustness is part of the spec. A design that meets the
mask only at its nominal etch is not a design; it is a coincidence. Gomez et
al. (Sci. Rep. 15, 2025) laser-ablated a pixelated notch filter and still
measured a 160 MHz centre shift from substrate and cutting tolerance, on a
device class whose features are finer than a stub's. So the number we report
per arm is the WORST CASE over the etch window, not the nominal.

The three-field formulation
---------------------------
This is the standard robust formulation from density-based topology
optimization (Wang, Lazarov & Sigmund, SMO 43:767-784, 2011): evaluate an
ERODED, a NOMINAL and a DILATED realization of the same design and optimize
(or here, report) the worst of the three. In the density setting the three
fields come from projecting one filtered field at three Heaviside thresholds
eta_e > eta_0 > eta_d, which for a filtered field is *approximately* a uniform
boundary offset.

Our evaluation happens downstream of thresholding -- the design has already
been binarized and rebuilt as hard-PEC ``Box`` geometry for the tier-1
imperative extractor (``xval1_imperative.py``) -- so we apply the boundary
offset DIRECTLY, as exact binary morphology on the cell mask. That is both
simpler and tighter than the projection proxy: erosion by n cells moves every
metal boundary inward by exactly n cells, which is what an over-etch does.

    over-etch  -> every metal feature SHRINKS  -> ``erode``
    nominal    -> as drawn                     -> the mask itself
    under-etch -> every metal feature GROWS    -> ``dilate``

Note the factor of two that trips people up: eroding by n cells removes n
cells from EACH side of a feature, so a strip loses 2n cells of width. That is
correct -- both edges move inward under an over-etch.

Connectivity: 4 (von Neumann / diamond structuring element)
-----------------------------------------------------------
``erode``/``dilate`` use the 5-point 4-connected structuring element, iterated
``cells`` times (iterating the diamond SE n times is exactly the L1 ball of
radius n, so the operation is exact, not approximate). The choice is
deliberate and is used consistently for both operations:

  * An isotropic wet/plasma etch offsets every boundary by the same EUCLIDEAN
    distance -- a disk structuring element. On a square grid the diamond (L1)
    and square (L_inf, 8-connected) SEs bracket the disk from inside and out.
  * The square SE retracts a 45-degree corner by n*sqrt(2) cells of Euclidean
    distance, i.e. it over-etches diagonal corners by 41 %. The diamond SE
    under-etches them by the same factor but is exact along the axes.
  * Microstrip layout on this fixture is Manhattan: the trace, the stubs and
    the design-box edges are all axis-aligned, so the axial directions are
    where nearly all of the metal boundary lives. The diamond SE is exact
    there, and the disagreement is confined to the corner cells of a free-form
    design.
  * Both SEs kill a 1-cell strip and close a 1-cell gap, which is the property
    this whole module exists to test, so that behaviour is not what decides the
    choice.

``connectivity=8`` is available for a sensitivity check and is strictly more
aggressive (it erodes more and dilates more). If a design's worst-case M moves
materially between the two, say so when reporting it -- that is a design whose
margin is being set by diagonal corner cells, which is itself a fragility
finding.

Boundary convention: PER OPERATION, PER SIDE, and never defaulted
-----------------------------------------------------------------
A morphological operation on a finite array has to assume something about the
cells just outside it, and on this fixture that assumption is load-bearing
physics rather than bookkeeping. The design box has four edges and they are not
alike:

  * the TRACE-ADJACENT edge abuts the fixed through-line. It is METAL, and it
    stays metal under an over-etch -- the etch does not open a seam between a
    stub and the line it is rooted on. Assume background there and every
    trace-rooted feature is spuriously cut free from the feed.
  * the OUTER transverse edge and BOTH along-line edges face bare dielectric.
    They are BACKGROUND. Assume metal there and a pad pushed hard against a box
    wall -- exactly what a bounded-box objective produces -- is never attacked
    by the over-etch on the side that touches the wall, which overstates its
    margin precisely where the design is most fragile.

Which array edge is the trace-adjacent one depends on the SIDE of the
two-sided box: masks are indexed in ascending GLOBAL y on both sides, so the
``hi`` (+y) side is rooted at j = 0 (``y_lo``) and the ``lo`` (-y) side is
rooted at j = ny-1 (``y_hi``). Getting this backwards detaches every stub on
one of the two sides. ``phase2_fixture.BoxSide.etch_outside`` is the single
place that decision is made; :func:`erode` and :func:`three_fields` take
``outside`` as a REQUIRED keyword so no call site can inherit a wrong default.

The two operations do NOT share the convention:

  * EROSION uses the four-edge tuple above (metal only on the trace edge).
  * DILATION assumes BACKGROUND on all four edges and therefore takes no
    ``outside`` argument at all. Growing metal in from the trace edge would
    widen the through-line by a cell along the whole box -- fusing every stub
    root into one continuous strip -- and trace widening/narrowing is
    explicitly out of scope for this module (see below). Growth is also clipped
    at the other three edges because the spec bounds all design metal to the
    box.

Calibrating the offset to real fabrication -- do the arithmetic in the open
---------------------------------------------------------------------------
Ordinary PCB etch tolerance on this class of board is about +-50 um of edge
placement (over-etch and under-etch both). The meshes in play:

    coarse cell  dx = 127.0 um   (the validated production fixture)
    refined cell dx =  63.5 um   (dx/2, the mesh-transferability check)

The morphology quantum is one cell, so:

  * REFINED mesh, cells=1: offset = 63.5 um vs a 50 um requirement.
    Ratio 63.5/50 = 1.27 -> CONSERVATIVE by 27 %. This is the nearest
    representable offset and it errs in the safe direction. **This is the
    default and it is the only combination in which a tolerance claim should
    be quoted.**
  * COARSE mesh, cells=1: offset = 127.0 um vs 50 um. Ratio 2.54 ->
    OVER-conservative by 2.5x. It tests a board nobody would ship. A design
    that fails here has not been shown to fail at spec tolerance, and saying
    so would be an overclaim in the pessimistic direction.
  * COARSE mesh, cells=0: offset = 0 um. That is not a tolerance test at all;
    it is the nominal field three times over. OPTIMISTIC -- it asserts perfect
    fabrication. Never report this as a robustness result.
  * REFINED mesh, cells=2: offset = 127.0 um. Same 2.54x over-conservatism as
    the coarse mesh at 1 cell; useful only as a stress case.

So: **the coarse mesh cannot express this tolerance.** It has no representable
offset between "nothing" and "2.5x too much". Descent may run on the coarse
mesh (Stage-2 plan, 45-period window), but the robust evaluation belongs on
the refined mesh, and that is also where the mesh-transferability gate
(PLAN gate 3) already sends every final design. The two requirements point the
same way, which is convenient rather than accidental.

Minimum feature size implied by the test
----------------------------------------
A feature of width w cells survives erosion by n cells iff w >= 2n+1. At the
default (refined mesh, n=1) that is 3 cells = 190.5 um, for both metal widths
and gaps. Any free-form design carrying 1- or 2-cell features is, by
construction, not manufacturable at this tolerance -- the erosion field will
show it as a collapsed structure and the worst-case M will say so. This is the
same minimum-length-scale quantity the TO literature imposes during
optimization (Lu, Wadbro, Berggren & Hassan, EuCAP 2025, use explicit
minimum-size control for exactly this reason); here it is measured after the
fact rather than enforced, because the point of the phase is to MEASURE margin,
not to manufacture it.

What this module does NOT model
-------------------------------
Deliberately out of scope, and each one is a reason the reported margin is an
upper bound on real robustness rather than the whole story:
  * substrate permittivity tolerance (RO4350B eps_r 3.66 +-0.05 -> ~0.7 %
    in eps_eff -> tens of MHz on its own),
  * conductor thickness / plating variation and trapezoidal etch profile
    (the sidewall is not vertical; we offset a 2-D footprint),
  * the through-line itself narrowing under the same etch (a passband and
    impedance effect, not a mask-morphology one),
  * registration error between layers (single-layer fixture, so nil here).

Run the self-test:  python research/metal_to/robust_eval.py
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field as dc_field

import numpy as np

from score_dualband import (  # the FROZEN metric -- imported, never redefined
    BAND_L_MHZ,
    BAND_U_MHZ,
    RELAXED,
    SCORE,
    Result,
    Thresholds,
    Validity,
    score,
    scoring_grid_mhz,
)

__all__ = [
    "erode", "dilate", "three_fields", "min_feature_cells",
    "EtchCalibration", "calibrate_etch",
    "FIELD_ORDER", "RobustResult", "robust_score", "robust_rank_key",
    "tolerance_report", "notch_centre_mhz",
    "OUTSIDE_TRACE_AT_Y_LO", "OUTSIDE_TRACE_AT_Y_HI", "OUTSIDE_DILATE",
    "Validity",                           # re-exported: callers build a
                                          # Validity per field and pass it in
]

# ---------------------------------------------------------------------------
# 1. Fabrication calibration constants  (arithmetic is in the module docstring)
# ---------------------------------------------------------------------------
DEFAULT_TOLERANCE_UM = 50.0     # ordinary PCB etch edge-placement tolerance
CELL_COARSE_UM = 127.0          # production fixture dx (msl_stub_notch_tuning.DX)
CELL_FINE_UM = 63.5             # dx/2, the mesh-transferability mesh
DEFAULT_CELL_UM = CELL_FINE_UM  # robustness is only quotable on the fine mesh
DEFAULT_CELLS = 1               # -> 63.5 um, 1.27x conservative vs 50 um

# Above this ratio the offset is so far past the spec tolerance that a failure
# says nothing about a manufacturable board. 2.0 admits the 1.27x default with
# room, and excludes the 2.54x coarse-mesh case.
CONSERVATISM_OVERSHOOT = 2.0

# Below this the test is not testing anything.
CONSERVATISM_UNDERSHOOT = 1.0

FIELD_ORDER = ("eroded", "nominal", "dilated")

# Structuring elements. Offsets are (di, dj) on a mask indexed [i_x, j_y]:
# i runs ALONG the line, j runs TRANSVERSE, matching the ``hard`` mask layout
# in xval1_imperative.mask_to_boxes.
_SE = {
    4: ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)),
    8: tuple((di, dj) for di in (-1, 0, 1) for dj in (-1, 0, 1)),
}


# ---------------------------------------------------------------------------
# 2. Exact binary morphology
# ---------------------------------------------------------------------------
def _as_bool_2d(mask) -> np.ndarray:
    m = np.asarray(mask)
    if m.ndim != 2:
        raise ValueError(f"mask must be 2-D (i_x, j_y); got shape {m.shape}")
    if m.dtype != bool:
        m = m >= 0.5           # thresholded design; gray input is a caller bug
    return m


def _norm_outside(outside) -> tuple[int, int, int, int]:
    """Value assumed for cells OUTSIDE the array, as (x_lo, x_hi, y_lo, y_hi).

    This is not a formality -- it decides whether a feature that touches the
    design-box boundary is cut free by the etch, and whether one pressed
    against a box wall is attacked there at all. See the module docstring's
    boundary-convention section; the per-side choice is made exactly once, in
    ``phase2_fixture.BoxSide.etch_outside``.

    A scalar is accepted only for the two unambiguous uniform cases (all
    background / all metal), which the duality and identity checks need.
    """
    if np.isscalar(outside):
        v = int(bool(outside))
        return (v, v, v, v)
    o = tuple(int(bool(v)) for v in outside)
    if len(o) != 4:
        raise ValueError("outside must be a scalar or (x_lo, x_hi, y_lo, y_hi)")
    return o


#: EROSION convention for a design side whose TRACE-ADJACENT row is j = 0,
#: i.e. the ``hi`` (+y) side of the two-sided box: metal beyond ``y_lo``,
#: bare dielectric beyond the other three edges.
OUTSIDE_TRACE_AT_Y_LO = (0, 0, 1, 0)

#: EROSION convention for a design side whose TRACE-ADJACENT row is j = ny-1,
#: i.e. the ``lo`` (-y) side, because masks are indexed in ascending GLOBAL y
#: on BOTH sides. Using :data:`OUTSIDE_TRACE_AT_Y_LO` here detaches every
#: lo-side stub from the feed line (measured: a 5-cell root -> 0 cells).
OUTSIDE_TRACE_AT_Y_HI = (0, 0, 0, 1)

#: DILATION convention, fixed and not a caller's choice: background on all
#: four edges. Metal must not grow in from the trace edge (that widens the
#: through-line, which this module declares out of scope, and fuses every stub
#: root into one strip) and must not grow out past the bounded box.
OUTSIDE_DILATE = (0, 0, 0, 0)


def _step(m: np.ndarray, outside, connectivity: int, grow: bool) -> np.ndarray:
    x_lo, x_hi, y_lo, y_hi = _norm_outside(outside)
    p = np.pad(m, 1, mode="constant",
               constant_values=((x_lo, x_hi), (y_lo, y_hi)))
    n_i, n_j = m.shape
    out = None
    for di, dj in _SE[connectivity]:
        # p is padded by 1, so index (1+i+di) reads neighbour (i+di) of m
        sl = p[1 + di:1 + di + n_i, 1 + dj:1 + dj + n_j]
        out = sl.copy() if out is None else (out | sl if grow else out & sl)
    return out


def _morph(mask, cells: int, outside, connectivity: int, grow: bool):
    """The one morphology primitive. ``outside`` is explicit and mandatory.

    :func:`erode` and :func:`dilate` are thin, correctly-parameterised wrappers
    around this; nothing else in the codebase implements binary morphology.
    """
    if cells < 0:
        raise ValueError("cells must be >= 0")
    if connectivity not in _SE:
        raise ValueError(f"connectivity must be 4 or 8, got {connectivity!r}")
    m = _as_bool_2d(mask)
    for _ in range(int(cells)):
        m = _step(m, outside, connectivity, grow=grow)
    return m


def erode(mask, cells: int = DEFAULT_CELLS, *, outside,
          connectivity: int = 4):
    """Shrink every metal feature by ``cells`` cells -- the OVER-ETCH field.

    Exact binary morphological erosion by the ``connectivity``-connected
    structuring element iterated ``cells`` times (= the L1 ball of radius
    ``cells`` for connectivity 4, the L_inf ball for 8).

    A metal feature of width w cells survives iff w >= 2*cells+1, because both
    of its edges move inward. A 1-cell strip therefore VANISHES at cells=1,
    which is the point: that is the fragile feature a topology optimizer likes
    to produce and it is not manufacturable.

    ``outside`` is REQUIRED and keyword-only: the assumed value of cells beyond
    the array, as (x_lo, x_hi, y_lo, y_hi). There is no default, because both
    plausible defaults are wrong in a way that silently flatters a design --
    all-background detaches every trace-rooted stub from the feed, all-metal
    exempts a wall-hugging pad from the over-etch on the wall side. Pass
    ``phase2_fixture.BoxSide.etch_outside`` (or the module constants
    :data:`OUTSIDE_TRACE_AT_Y_LO` / :data:`OUTSIDE_TRACE_AT_Y_HI`), chosen per
    SIDE of the two-sided box.
    """
    return _morph(mask, cells, outside, connectivity, grow=False)


def dilate(mask, cells: int = DEFAULT_CELLS, *, connectivity: int = 4):
    """Grow every metal feature by ``cells`` cells -- the UNDER-ETCH field.

    Exact binary morphological dilation with the same structuring element as
    :func:`erode`; the self-test asserts the De Morgan duality
    ``dilate(m) == ~erode(~m, outside=1)`` holds exactly, boundary included.

    A gap of width w cells CLOSES iff w <= 2*cells, because both of its walls
    move inward. A 1-cell gap therefore closes at cells=1 -- the other fragile
    free-form feature, and the one that turns two separate resonators into one
    merged notch, which on this spec is precisely the S_G failure.

    There is deliberately NO ``outside`` argument. Dilation assumes BACKGROUND
    on all four edges (:data:`OUTSIDE_DILATE`):

      * on the trace-adjacent edge, ORing in the metal through-line would set
        the ENTIRE trace-adjacent row of the box -- on the production box a
        94-cell (11.9 mm) strip shorting every stub root together -- and it
        would model the through-line WIDENING under the under-etch while the
        over-etch field does not model it narrowing. Trace width variation is
        out of scope for this module in both directions, not just one.
      * on the other three edges, the spec bounds all design metal to the box,
        so the box wall is a hard constraint rather than a physical feature; a
        design that only survives by growing past it is out of spec at nominal
        already.
    """
    return _morph(mask, cells, OUTSIDE_DILATE, connectivity, grow=True)


def three_fields(mask, cells: int = DEFAULT_CELLS, *, outside,
                 connectivity: int = 4) -> dict:
    """The eroded / nominal / dilated realizations of one binary design.

    Returns a dict keyed by :data:`FIELD_ORDER`. ``outside`` is REQUIRED and
    keyword-only for the reason spelled out in :func:`erode`; it reaches the
    erosion only, since :func:`dilate` fixes its own convention.

    Feed each mask through the SAME geometry-rebuild and solve path
    (``phase2_fixture.boxes_from_mask`` / ``xval1_imperative.mask_to_boxes`` ->
    ``compute_msl_s_matrix``) so the three scores differ only by the etch.
    """
    m = _as_bool_2d(mask)
    return {
        "eroded": erode(m, cells, outside=outside, connectivity=connectivity),
        "nominal": m.copy(),
        "dilated": dilate(m, cells, connectivity=connectivity),
    }


def min_feature_cells(cells: int = DEFAULT_CELLS) -> int:
    """Minimum metal width (and minimum gap) in cells that survives the test."""
    return 2 * int(cells) + 1


# ---------------------------------------------------------------------------
# 3. Calibration of the offset to a physical tolerance
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class EtchCalibration:
    tolerance_um: float      # what fabrication actually guarantees
    cell_um: float           # mesh cell the design is evaluated on
    cells: int               # morphological offset, in cells
    connectivity: int = 4

    @property
    def realized_um(self) -> float:
        return self.cells * self.cell_um

    @property
    def conservatism(self) -> float:
        """realized / required. >1 = conservative, <1 = optimistic."""
        return self.realized_um / self.tolerance_um if self.tolerance_um else math.inf

    @property
    def min_feature_um(self) -> float:
        return min_feature_cells(self.cells) * self.cell_um

    @property
    def quotable(self) -> bool:
        return (CONSERVATISM_UNDERSHOOT <= self.conservatism
                <= CONSERVATISM_OVERSHOOT)

    @property
    def verdict(self) -> str:
        c = self.conservatism
        if self.cells == 0:
            return "NO TEST (offset 0 um) -- optimistic, not a robustness result"
        if c < CONSERVATISM_UNDERSHOOT:
            return f"OPTIMISTIC ({c:.2f}x) -- offset is below the spec tolerance"
        if c > CONSERVATISM_OVERSHOOT:
            return (f"OVER-CONSERVATIVE ({c:.2f}x) -- a failure here does not "
                    "demonstrate a failure at spec tolerance")
        return f"CONSERVATIVE ({c:.2f}x) -- quotable"

    def describe(self) -> str:
        return (f"+-{self.cells} cell x {self.cell_um:.1f} um = "
                f"+-{self.realized_um:.1f} um vs spec +-{self.tolerance_um:.0f} um "
                f"[{self.verdict}]; conn={self.connectivity}; "
                f"min feature {min_feature_cells(self.cells)} cells = "
                f"{self.min_feature_um:.1f} um")


def calibrate_etch(tolerance_um: float = DEFAULT_TOLERANCE_UM,
                   cell_um: float = DEFAULT_CELL_UM,
                   cells: int | None = None,
                   connectivity: int = 4) -> EtchCalibration:
    """Pick the morphological offset for a mesh and a fabrication tolerance.

    With ``cells=None`` the offset is the smallest integer number of cells that
    COVERS the tolerance (ceil), so the default errs conservative. On the fine
    mesh (63.5 um) that is 1 cell = 63.5 um for a 50 um tolerance. Pass
    ``cells`` explicitly to force a stress case, then read ``.verdict``.
    """
    if cell_um <= 0:
        raise ValueError("cell_um must be > 0")
    n = int(math.ceil(tolerance_um / cell_um)) if cells is None else int(cells)
    return EtchCalibration(tolerance_um=float(tolerance_um),
                           cell_um=float(cell_um), cells=n,
                           connectivity=int(connectivity))


# ---------------------------------------------------------------------------
# 4. Robust scoring across the three fields
# ---------------------------------------------------------------------------
TERMS = ("S_L", "S_U", "S_G", "S_P")

_TERM_MEANING = {
    "S_L": "lower WLAN band under-rejected",
    "S_U": "upper WLAN band under-rejected",
    "S_G": "inter-band gap blocked (notches merged)",
    "S_P": "passband insertion loss",
}


def _search_windows(band_l, band_u, pad_mhz):
    """One tracking window per notch, split at the inter-band midpoint.

    Widening each stopband by ``pad`` lets a drifting notch be followed, but a
    naive pad crosses into the neighbouring band and then both windows report
    the same (deeper) notch. Clipping at the midpoint keeps each notch tracked
    on its own side, which is the only assignment that stays meaningful when a
    design's two notches have moved.
    """
    mid = 0.5 * (band_l[1] + band_u[0])
    return ((band_l[0] - pad_mhz, min(band_l[1] + pad_mhz, mid)),
            (max(band_u[0] - pad_mhz, mid), band_u[1] + pad_mhz))


def notch_centre_mhz(freqs_mhz, il_db, window):
    """Free-running notch centre: argmax of IL inside ``window`` (lo, hi) MHz.

    ``score_dualband.Result.f_notch_*`` searches only INSIDE the stopband, so a
    notch that drifts out of the band is reported PINNED at the band edge and
    its true movement is hidden. For a tolerance report the drift is the
    quantity of interest, so pass a widened window (see :func:`_search_windows`).

    Interpretation limit from the frozen metric still applies: never quote a
    centre or a shift to better than +-50 MHz.
    """
    f = np.asarray(freqs_mhz, dtype=float)
    y = np.asarray(il_db, dtype=float)
    m = (f >= window[0]) & (f <= window[1])
    if not m.any():
        return float("nan")
    return float(f[m][int(np.argmax(y[m]))])


@dataclass
class RobustResult:
    """Worst-case-over-etch summary. ``M_worst`` is the number to report."""
    M_worst: float
    M_nominal: float
    M_best: float
    M_spread: float
    worst_field: str
    Omega_worst: float
    Omega_nominal: float
    spec_pass_all: bool
    spec_pass_nominal: bool
    degenerate_any: bool
    all_valid: bool | None
    per_field: dict                       # name -> score_dualband.Result
    term_deltas: dict                     # name -> {term: delta vs nominal}
    notch_shift_mhz: dict                 # name -> {"L": d, "U": d}
    etch: EtchCalibration | None = None
    notes: list = dc_field(default_factory=list)

    @property
    def fragility_db(self) -> float:
        """How much of the margin the etch eats: M_worst - M_nominal."""
        return self.M_worst - self.M_nominal


# Per-field keys a response dict may carry: they describe THAT realization's
# measured response and nothing else.
_PER_FIELD_KEYS = ("s11_db", "s21_db_abs", "validity")

# Keys that define the SCORING PROBLEM rather than a measurement. They are set
# once for the whole three-field comparison and are refused per field: the
# fields must be scored against one identical mask, and ``_search_windows``
# reads the top-level ``band_l``/``band_u`` regardless, so a per-field override
# would silently score a field on one band pair and track its notch drift on
# another. Pass them to ``robust_score`` instead, where they apply to all three.
_WHOLE_COMPARISON_KEYS = ("band_l", "band_u", "guard_mhz", "f_lo", "f_hi")


def _coerce_result(v, thr: Thresholds, score_kwargs: dict):
    """Accept an already-scored ``Result`` or a raw response dict/tuple."""
    if isinstance(v, Result):
        return v, None
    if isinstance(v, dict):
        d = dict(v)
        f = d.pop("freqs_mhz", None)
        il = d.pop("il_db", None)
        if f is None or il is None:
            raise ValueError("response dict needs 'freqs_mhz' and 'il_db'")
        bad = [k for k in _WHOLE_COMPARISON_KEYS if k in d]
        if bad:
            raise ValueError(
                f"per-field response may not override {sorted(bad)}: those "
                "define the scoring problem for the WHOLE three-field "
                "comparison. _search_windows would keep the top-level band "
                "pair while score() used the override, so the notch-drift "
                "windows and the scored bands would disagree. Pass them to "
                "robust_score() instead.")
        kw = dict(score_kwargs)
        for k in _PER_FIELD_KEYS:
            if k in d:
                kw[k] = d.pop(k)
        if d:
            raise ValueError(f"unrecognised response keys: {sorted(d)}")
        return score(f, il, thr=thr, **kw), (np.asarray(f), np.asarray(il))
    if isinstance(v, (tuple, list)) and len(v) == 2:
        f, il = v
        return score(f, il, thr=thr, **score_kwargs), (np.asarray(f),
                                                       np.asarray(il))
    raise TypeError(f"cannot interpret field response of type {type(v)!r}")


def robust_score(per_field_responses, thr: Thresholds = SCORE,
                 etch: EtchCalibration | None = None,
                 nominal_field: str = "nominal",
                 band_l=BAND_L_MHZ, band_u=BAND_U_MHZ,
                 search_pad_mhz: int = 400,
                 **score_kwargs) -> RobustResult:
    """Combine the three etch fields into one worst-case margin.

    Parameters
    ----------
    per_field_responses : mapping ``{"eroded"|"nominal"|"dilated": response}``.
        A response is either an already-computed ``score_dualband.Result``, a
        dict ``{"freqs_mhz":..., "il_db":..., optional "s11_db",
        "s21_db_abs", "validity"}``, or a bare ``(freqs_mhz, il_db)`` pair.
        Each response is scored with ``score_dualband.score`` UNCHANGED.
    thr : the frozen ``SCORE`` thresholds (or ``RELAXED`` for the arm-D
        benchmark-kill gate). Not redefined here.
    etch : the calibration used to build the fields, carried through for the
        report so a number can never be quoted without its offset.

    Returns
    -------
    RobustResult. ``M_worst = max(M)`` over the fields, because ``M`` is
    lower-is-better; ``Omega_worst = min(Omega)`` for the same reason. Those
    two are the margin the comparison reports. ``M_nominal`` and ``M_spread``
    are kept alongside so the cost of robustness is visible rather than
    silently folded in.
    """
    missing = [k for k in FIELD_ORDER if k not in per_field_responses]
    if missing:
        raise ValueError(f"missing etch field(s): {missing}; "
                         f"the three-field formulation needs all of {FIELD_ORDER}")
    extra = [k for k in per_field_responses if k not in FIELD_ORDER]
    if extra:
        raise ValueError(f"unexpected field(s) {extra}; expected {FIELD_ORDER}")
    if nominal_field not in FIELD_ORDER:
        raise ValueError(f"nominal_field must be one of {FIELD_ORDER}")

    score_kwargs.setdefault("band_l", band_l)
    score_kwargs.setdefault("band_u", band_u)

    per_field, traces = {}, {}
    for name in FIELD_ORDER:
        r, tr = _coerce_result(per_field_responses[name], thr, score_kwargs)
        per_field[name] = r
        traces[name] = tr

    nom = per_field[nominal_field]
    Ms = {n: per_field[n].M for n in FIELD_ORDER}
    worst_field = max(FIELD_ORDER, key=lambda n: Ms[n])

    term_deltas = {n: {t: getattr(per_field[n], t) - getattr(nom, t)
                       for t in TERMS} for n in FIELD_ORDER}

    win_l, win_u = _search_windows(band_l, band_u, search_pad_mhz)
    shifts = {}
    for n in FIELD_ORDER:
        if traces.get(n) is not None and traces.get(nominal_field) is not None:
            f_n, il_n = traces[n]
            f_0, il_0 = traces[nominal_field]
            shifts[n] = {
                "L": (notch_centre_mhz(f_n, il_n, win_l)
                      - notch_centre_mhz(f_0, il_0, win_l)),
                "U": (notch_centre_mhz(f_n, il_n, win_u)
                      - notch_centre_mhz(f_0, il_0, win_u)),
            }
        else:
            # only pre-scored Results were supplied: fall back to the frozen
            # metric's in-band centres, which SATURATE at the band edges.
            shifts[n] = {
                "L": per_field[n].f_notch_L_MHz - nom.f_notch_L_MHz,
                "U": per_field[n].f_notch_U_MHz - nom.f_notch_U_MHz,
            }

    # score() stores validity as ``asdict(Validity)``, and ``Validity.ok`` is a
    # PROPERTY -- asdict does not carry it. Rebuild the dataclass and ask it,
    # rather than looking for an "ok" key that is never there (a check that
    # would silently pass every time).
    vals = [r.validity for r in per_field.values()]
    if all(v is None for v in vals):
        all_valid = None
    else:
        all_valid = all(Validity(**v).ok if isinstance(v, dict)
                        else bool(v is not None and v.ok) for v in vals)
    invalid_fields = ([] if all_valid is not False else
                      [n for n in FIELD_ORDER
                       if per_field[n].validity is not None
                       and not Validity(**per_field[n].validity).ok])

    notes = []
    if etch is not None and not etch.quotable:
        notes.append(f"etch calibration not quotable: {etch.verdict}")
    if all_valid is False:
        notes.append("field(s) " + ", ".join(invalid_fields) + " failed a "
                     "score_dualband validity gate -- NOT quotable")
    if traces.get(nominal_field) is None:
        notes.append("notch shifts fall back to the frozen metric's IN-BAND "
                     "centres and saturate at the band edges; pass raw traces "
                     "for free-running drift")

    return RobustResult(
        M_worst=float(Ms[worst_field]),
        M_nominal=float(nom.M),
        M_best=float(min(Ms.values())),
        M_spread=float(max(Ms.values()) - min(Ms.values())),
        worst_field=worst_field,
        Omega_worst=float(min(per_field[n].Omega for n in FIELD_ORDER)),
        Omega_nominal=float(nom.Omega),
        spec_pass_all=bool(all(per_field[n].spec_pass for n in FIELD_ORDER)),
        spec_pass_nominal=bool(nom.spec_pass),
        degenerate_any=bool(any(per_field[n].degenerate for n in FIELD_ORDER)),
        all_valid=all_valid,
        per_field=per_field, term_deltas=term_deltas, notch_shift_mhz=shifts,
        etch=etch, notes=notes,
    )


def robust_rank_key(rr: RobustResult):
    """Ranking across arms, mirroring ``score_dualband.rank_key`` but on the
    WORST case: degenerate last, then M_worst ascending, then Omega_worst
    descending. Under the PI's margin framing this is what orders the arms."""
    return (rr.degenerate_any, rr.M_worst, -rr.Omega_worst)


# ---------------------------------------------------------------------------
# 5. Human-readable report
# ---------------------------------------------------------------------------
def _driver(deltas: dict) -> str:
    t = max(TERMS, key=lambda k: deltas[k])
    if deltas[t] <= 1e-9:
        return "none (no term degrades)"
    return f"{t} +{deltas[t]:.2f}  ({_TERM_MEANING[t]})"


def tolerance_report(rr: RobustResult, design: str = "design",
                     width: int = 96) -> str:
    """Compact table: per-field terms, worst case, degradation drivers, drift.

    Returns the text (also printable). Shifts smaller than the 100 MHz record
    resolution are marked ``~`` -- the frozen metric forbids quoting a notch
    centre better than +-50 MHz, so those are indicative only.
    """
    L = []
    rule = "-" * width
    L.append(rule)
    L.append(f"Etch-tolerance robustness (three-field)  |  design: {design}")
    L.append("metric: score_dualband.score, frozen and unmodified. LOWER M is better; M=0 meets the mask.")
    if rr.etch is not None:
        L.append(f"etch:   {rr.etch.describe()}")
    L.append(rule)
    L.append(f"{'field':9s} {'M':>7s} {'S_L':>6s} {'S_U':>6s} {'S_G':>6s} "
             f"{'S_P':>6s} {'Omega':>7s} {'f_L':>6s} {'df_L':>7s} "
             f"{'f_U':>6s} {'df_U':>7s}  flags")
    for n in FIELD_ORDER:
        r = rr.per_field[n]
        s = rr.notch_shift_mhz[n]
        flags = []
        if n == rr.worst_field:
            flags.append("WORST")
        if r.spec_pass:
            flags.append("pass")
        if r.degenerate:
            flags.append("DEGENERATE")
        def _d(x):
            if not np.isfinite(x):
                return "   n/a"
            return f"{x:+6.0f}" + ("~" if abs(x) < 50 else " ")
        L.append(f"{n:9s} {r.M:7.2f} {r.S_L:6.2f} {r.S_U:6.2f} {r.S_G:6.2f} "
                 f"{r.S_P:6.2f} {r.Omega:7.2f} "
                 f"{r.f_notch_L_MHz:6.0f} {_d(s['L'])} "
                 f"{r.f_notch_U_MHz:6.0f} {_d(s['U'])}  {' '.join(flags)}")
    L.append(rule)
    L.append(f"WORST-CASE M = {rr.M_worst:6.2f}   (field: {rr.worst_field})"
             f"      nominal M = {rr.M_nominal:6.2f}"
             f"      spread = {rr.M_spread:5.2f}")
    L.append(f"margin       Omega_worst = {rr.Omega_worst:+6.2f} dB"
             f"       Omega_nominal = {rr.Omega_nominal:+6.2f} dB"
             f"       fragility = {rr.fragility_db:+5.2f} dB of M")
    L.append(f"driver  over-etch  (eroded) : {_driver(rr.term_deltas['eroded'])}")
    L.append(f"driver  under-etch (dilated): {_driver(rr.term_deltas['dilated'])}")
    L.append(f"spec: nominal {'PASS' if rr.spec_pass_nominal else 'fail'}"
             f"   |   all three fields "
             f"{'PASS' if rr.spec_pass_all else 'fail'}"
             + (f"   |   validity {'ok' if rr.all_valid else 'FAILED'}"
                if rr.all_valid is not None else ""))
    L.append("f_L/f_U are the frozen metric's IN-BAND centres and pin at a band edge if a notch "
             "leaves its band;")
    L.append("df is the free-running drift on a widened window. '~' marks a shift below the "
             "100 MHz record resolution.")
    for note in rr.notes:
        L.append(f"NOTE: {note}")
    L.append(rule)
    return "\n".join(L)


# ---------------------------------------------------------------------------
# 6. Self-test
# ---------------------------------------------------------------------------
def _fmt_mask(m, title):
    out = [title]
    for j in range(m.shape[1] - 1, -1, -1):          # print +y upward
        out.append("   " + "".join("#" if m[i, j] else "." for i in range(m.shape[0])))
    return "\n".join(out)


def _il_bandstop(f_mhz, sections, floor_db=0.2):
    """SYNTHETIC response: a cascade of order-N Butterworth bandstop sections.

    NOT a solve and not a claim about any arm. It exists only so the reporting
    plumbing can be exercised against traces that have the right SHAPE.

    Each section is ``(f0_MHz, bw20_MHz, depth_dB, order)``. With the standard
    bandstop-to-lowpass mapping u = (f/f0 - f0/f)/FBW,

        IL(f) = 10*log10(1 + 1/(u^(2N) + eps)),   eps = 1/(10^(depth/10) - 1)

    and FBW is solved so that IL = 20 dB exactly at f0 +- bw20/2.

    The order matters and is the reason this replaced a single-pole model: with
    N = 1 a section whose 20-dB width covers the 200 MHz lower WLAN band also
    puts ~15 dB into the 5450-5625 gap, so NO single-pole pair can satisfy this
    mask. That is the benchmark's actual difficulty (NOTE_stage0_window.md,
    correction 3) rather than a modelling artifact -- the mask needs roughly
    four to five resonators per band, which is precisely what the bounded box
    is suspected not to admit. N = 4 is used below so the synthetic designs sit
    near the mask and the etch decides the outcome, which is what a report
    fixture needs to show.
    """
    f = np.asarray(f_mhz, dtype=float)
    lin = np.ones_like(f)
    for f0, bw20, depth, order in sections:
        f_edge = f0 + bw20 / 2.0
        u_edge = f_edge / f0 - f0 / f_edge
        fbw = u_edge / 99.0 ** (-1.0 / (2.0 * order))
        u = np.abs((f / f0 - f0 / f) / fbw)
        eps = 1.0 / (10.0 ** (depth / 10.0) - 1.0)
        il = 10 * np.log10(1.0 + 1.0 / (u ** (2 * order) + eps))
        lin *= 10 ** (-il / 10.0)
    return -10 * np.log10(lin) + floor_db


def _selftest() -> int:
    print("=" * 96)
    print("robust_eval self-test -- morphology first, then the report plumbing")
    print("=" * 96)

    cal = calibrate_etch()
    print("\n[calibration]")
    print(f"  default (refined mesh): {cal.describe()}")
    for label, cell, n in (("coarse mesh, 1 cell", CELL_COARSE_UM, 1),
                           ("coarse mesh, 0 cells", CELL_COARSE_UM, 0),
                           ("refined mesh, 2 cells", CELL_FINE_UM, 2)):
        c = calibrate_etch(cell_um=cell, cells=n)
        print(f"  {label:22s}: {c.describe()}")
    assert cal.cells == 1 and abs(cal.realized_um - 63.5) < 1e-9
    assert cal.quotable, "the default combination must be the quotable one"
    assert not calibrate_etch(cell_um=CELL_COARSE_UM, cells=1).quotable, \
        "coarse mesh at 1 cell is 2.54x and must be flagged unquotable"
    assert not calibrate_etch(cell_um=CELL_COARSE_UM, cells=0).quotable, \
        "zero offset is not a robustness test"
    assert min_feature_cells(1) == 3
    print(f"  -> minimum manufacturable feature at the default: "
          f"{min_feature_cells(cal.cells)} cells = {cal.min_feature_um:.1f} um "
          f"(metal width AND gap)")

    # ---- (a) a 1-cell strip must VANISH under erosion -------------------
    print("\n[a] 1-cell-wide strip: the feature a topology optimizer loves")
    strip = np.zeros((9, 7), dtype=bool)
    strip[4, 1:6] = True                      # 1 cell wide (x), 5 long (y)
    print(_fmt_mask(strip, "  nominal (width 1):"))
    e1 = erode(strip, 1, outside=OUTSIDE_TRACE_AT_Y_LO)
    print(_fmt_mask(e1, "  eroded by 1 cell:"))
    assert not e1.any(), "1-cell strip must vanish entirely under 1-cell erosion"
    print(f"  -> metal cells {int(strip.sum())} -> {int(e1.sum())}: VANISHED. "
          "A design that needs this strip does not exist on a real board.")

    # width 2 also dies (both edges move in); width 3 survives as width 1
    for w, alive in ((1, False), (2, False), (3, True), (5, True)):
        s = np.zeros((11, 7), dtype=bool)
        s[4:4 + w, 1:6] = True
        got = erode(s, 1, outside=OUTSIDE_TRACE_AT_Y_LO).any()
        assert got == alive, f"width {w}: expected survive={alive}"
    print("  -> width sweep: 1,2 vanish; 3,5 survive. Matches w >= 2n+1 = 3.")

    # ---- (b) a 1-cell gap must CLOSE under dilation ----------------------
    print("\n[b] 1-cell gap: the other fragile free-form feature")
    gap = np.zeros((9, 7), dtype=bool)
    gap[1:4, 1:6] = True
    gap[5:8, 1:6] = True                      # column 4 is a 1-cell gap
    print(_fmt_mask(gap, "  nominal (gap at column 4):"))
    d1 = dilate(gap, 1)
    print(_fmt_mask(d1, "  dilated by 1 cell:"))
    assert gap[4, 1:6].sum() == 0, "fixture: the gap must start open"
    assert d1[4, 1:6].all(), "1-cell gap must close under 1-cell dilation"
    print("  -> the gap is BRIDGED: two separate resonators become one. "
          "On this spec that is the S_G failure (merged notch, blocked gap).")
    for g, closes in ((1, True), (2, True), (3, False)):
        m = np.zeros((13, 7), dtype=bool)
        m[1:5, 1:6] = True
        m[5 + g:9 + g, 1:6] = True
        dd = dilate(m, 1)
        assert bool(dd[5:5 + g, 3].all()) == closes, f"gap {g}: expected close={closes}"
    print("  -> gap sweep: 1,2 close; 3 survives. Matches gap >= 2n+1 = 3.")

    # ---- (c) boundary convention -- per side, per operation --------------
    #
    # REGRESSION GATES. Every assertion below is a number a joint review
    # measured on the two implementations this module replaced, and each
    # comment carries the BROKEN value alongside the correct one so a
    # regression has to be re-measured rather than re-argued. The broken value
    # is recomputed live from ``_morph`` with the old convention, not
    # hard-coded from the review, so these stay honest if the SE ever changes.
    print("\n[c] boundary convention (per side, per operation) -- regression gates")

    # The production two-sided box, per side (phase2_fixture.NX_BOX/NY_SIDE).
    # Duplicated as literals rather than imported: robust_eval must not pull in
    # rfx/jax to run its own self-test.
    NX_BOX, NY_SIDE = 94, 70

    # -- (c-i) 'hi' side: the trace-adjacent row is j = 0 -------------------
    print("\n  (i) 'hi' (+y) side: trace-adjacent row is j = 0 "
          "-> OUTSIDE_TRACE_AT_Y_LO")
    stub = np.zeros((11, 22), dtype=bool)
    stub[3:8, 0:20] = True                    # 5 wide, 20 long, rooted at j=0
    e = erode(stub, 1, outside=OUTSIDE_TRACE_AT_Y_LO)
    d = dilate(stub, 1)
    w_e = int(e[:, 5].sum()); l_e = int(e[5, :].sum())
    w_d = int(d[:, 5].sum()); l_d = int(d[5, :].sum())
    print(f"      nominal   width 5 cells, length 20 cells")
    print(f"      eroded    width {w_e} cells, length {l_e} cells  (both side "
          "edges move in; the ROOT survives, the TIP is attacked)")
    print(f"      dilated   width {w_d} cells, length {l_d} cells")
    assert (w_e, l_e) == (3, 19), (w_e, l_e)
    assert (w_d, l_d) == (7, 21), (w_d, l_d)
    assert int(e[:, 0].sum()) == 3, "hi-side stub ROOT must survive erosion"

    # D3: dilation must NOT fill the trace-adjacent row.
    row_d = int(d[:, 0].sum())
    row_d_broken = int(_morph(stub, 1, OUTSIDE_TRACE_AT_Y_LO, 4,
                              grow=True)[:, 0].sum())
    print(f"      dilated trace-adjacent row j=0: {row_d}/{stub.shape[0]} cells "
          f"(ORing the metal boundary in, as the old _step did, gave "
          f"{row_d_broken}/{stub.shape[0]} -- the WHOLE row)")
    assert row_d == 7, row_d
    assert not d[:, 0].all(), \
        "dilation must not fill the trace-adjacent row: that is the " \
        "through-line WIDENING, which this module declares out of scope"
    assert row_d_broken == stub.shape[0], \
        "fixture: the old grow-with-metal-boundary really did fill the row"

    # -- (c-ii) 'lo' side: the trace-adjacent row is j = ny-1 ---------------
    # Masks ascend in GLOBAL y on both sides, so the lo side is rooted at the
    # LAST column, not the first. Measured on the production box.
    print("\n  (ii) 'lo' (-y) side: trace-adjacent row is j = ny-1 "
          "-> OUTSIDE_TRACE_AT_Y_HI")
    lo = np.zeros((NX_BOX, NY_SIDE), dtype=bool)
    lo[44:49, NY_SIDE - 20:NY_SIDE] = True    # 5 wide, 20 deep, rooted at ny-1
    root_ok = int(erode(lo, 1, outside=OUTSIDE_TRACE_AT_Y_HI)[:, -1].sum())
    root_bad = int(erode(lo, 1, outside=OUTSIDE_TRACE_AT_Y_LO)[:, -1].sum())
    print(f"      root row (j = ny-1) after erosion: {root_ok} cells with the "
          f"lo convention, {root_bad} with the hi convention "
          f"(nominal 5) -- the hi convention DETACHES IT FROM THE FEED LINE")
    assert root_ok == 3, root_ok
    assert root_bad == 0, root_bad
    assert erode(lo, 1, outside=OUTSIDE_TRACE_AT_Y_HI)[:, -1].any(), \
        "a lo-side stub ROOT must SURVIVE erosion"

    row_lo = int(dilate(lo, 1)[:, -1].sum())
    row_lo_broken = int(_morph(lo, 1, OUTSIDE_TRACE_AT_Y_HI, 4,
                               grow=True)[:, -1].sum())
    print(f"      dilated trace-adjacent row: {row_lo} cells; ORing the metal "
          f"boundary in gave {row_lo_broken} = the full {NX_BOX}-cell "
          f"({NX_BOX * 127.0 / 1000:.1f} mm) box width, shorting every stub "
          f"root together")
    assert row_lo == 7 and row_lo_broken == NX_BOX, (row_lo, row_lo_broken)

    # -- (c-iii) a pad hard against the OUTER walls must be eroded there ----
    # The outer transverse edge and both along-line edges face bare dielectric.
    # ``shrink = ~dilate(~m)`` with clipped dilation is an erosion with metal
    # assumed on ALL FOUR edges, which exempts a wall-hugging pad from the
    # over-etch exactly where a bounded-box objective pushes metal.
    print("\n  (iii) 3x3 pad hard against x_lo AND the outer y edge")
    pad = np.zeros((NX_BOX, NY_SIDE), dtype=bool)
    pad[0:3, 0:3] = True                      # lo side: j=0 is the OUTER edge
    n_ok = int(erode(pad, 1, outside=OUTSIDE_TRACE_AT_Y_HI).sum())
    n_broken = int(_morph(pad, 1, (1, 1, 1, 1), 4, grow=False).sum())
    print(f"      9 cells -> {n_ok} under the correct erosion, "
          f"-> {n_broken} under 'outside = metal on all four edges'")
    assert n_ok == 1, n_ok
    assert n_broken == 4, n_broken
    assert not erode(pad, 1, outside=OUTSIDE_TRACE_AT_Y_HI)[0, :].any(), \
        "a pad against the outer box wall MUST be eroded at that wall"

    # -- (c-iv) a full-depth stub must lose its TIP ------------------------
    print("\n  (iv) full-depth stub: the tip touches the outer wall")
    deep = np.zeros((NX_BOX, NY_SIDE), dtype=bool)
    deep[44:49, :] = True                     # 70 cells deep, tip at j = 0
    len_ok = int(erode(deep, 1, outside=OUTSIDE_TRACE_AT_Y_HI)[46, :].sum())
    len_broken = int(_morph(deep, 1, (1, 1, 1, 1), 4, grow=False)[46, :].sum())
    print(f"      length {NY_SIDE} cells -> {len_ok} under the correct erosion "
          f"(tip attacked, root kept), -> {len_broken} under all-metal "
          f"(TIP ENTIRELY PRESERVED -- no over-etch at all)")
    assert (len_ok, len_broken) == (NY_SIDE - 1, NY_SIDE), (len_ok, len_broken)

    # 63.5 um off a ~8.5 mm stub is a ~0.75 % length change -> ~40 MHz at 5.25 GHz
    print(f"\n  -> physical meaning: 1 fine cell off a 20-cell (1.27 mm at "
          "63.5 um) stub is 5 % of its length; off a real 8.5 mm stub it is "
          "0.75 %, i.e. ~39 MHz at 5.25 GHz and ~43 MHz at 5.775 GHz.")

    # ---- (d) operator sanity: duality, monotonicity, composition ---------
    print("\n[d] operator properties (exactness checks)")
    rng = np.random.default_rng(20260827)
    for conn in (4, 8):
        for _ in range(200):
            m = rng.random((9, 8)) < rng.uniform(0.15, 0.85)
            out = tuple(int(x) for x in rng.integers(0, 2, 4))
            inv_out = tuple(1 - x for x in out)
            n = int(rng.integers(0, 3))
            # De Morgan duality at the PRIMITIVE level, boundary included, for
            # every one of the 16 boundary conventions
            assert np.array_equal(_morph(m, n, out, conn, grow=True),
                                  ~_morph(~m, n, inv_out, conn, grow=False)), \
                "duality"
            # the public dilate() is the primitive at OUTSIDE_DILATE, so it is
            # dual to an erosion that assumes METAL on all four edges
            assert np.array_equal(dilate(m, n, connectivity=conn),
                                  ~erode(~m, n, outside=1, connectivity=conn))
            # monotone containment (the SE contains (0,0), so this holds for
            # any boundary convention)
            assert (erode(m, n, outside=out, connectivity=conn) <= m).all()
            assert (m <= dilate(m, n, connectivity=conn)).all()
            # composition == iteration
            assert np.array_equal(
                erode(m, 2, outside=out, connectivity=conn),
                erode(erode(m, 1, outside=out, connectivity=conn), 1,
                      outside=out, connectivity=conn))
            assert np.array_equal(
                dilate(m, 2, connectivity=conn),
                dilate(dilate(m, 1, connectivity=conn), 1, connectivity=conn))
            # zero offset is the identity
            assert np.array_equal(erode(m, 0, outside=out, connectivity=conn), m)
            assert np.array_equal(dilate(m, 0, connectivity=conn), m)
    print("  -> duality, monotonicity, composition and identity hold exactly "
          "for connectivity 4 and 8, over 400 random masks x 16 boundary "
          "conventions.")

    # connectivity 8 is strictly more aggressive on a diagonal corner
    corner = np.zeros((7, 7), dtype=bool)
    for i in range(1, 6):
        corner[i, 1:7 - i] = True             # 45-degree staircase edge
    n4 = int(erode(corner, 1, outside=0, connectivity=4).sum())
    n8 = int(erode(corner, 1, outside=0, connectivity=8).sum())
    assert n8 <= n4
    print(f"  -> 45-degree staircase: erosion keeps {n4} cells at conn=4 vs "
          f"{n8} at conn=8. conn=8 is the more aggressive bound; conn=4 is the "
          "default because this fixture's metal is Manhattan.")

    # ---- (e) three_fields wiring ----------------------------------------
    print("\n[e] three_fields() on a design carrying BOTH fragile features")
    mixed = np.zeros((15, 10), dtype=bool)
    mixed[2:7, 0:8] = True                    # solid resonator, rooted
    mixed[7, 3] = True                        # 1-cell bridge
    mixed[8:13, 0:8] = True                   # second resonator
    mixed[13, 2:6] = True                     # 1-cell-wide spur
    tf = three_fields(mixed, cal.cells, outside=OUTSIDE_TRACE_AT_Y_LO)
    assert set(tf) == set(FIELD_ORDER)
    print(_fmt_mask(tf["eroded"], "  eroded:"))
    print(_fmt_mask(tf["nominal"], "  nominal:"))
    print(_fmt_mask(tf["dilated"], "  dilated:"))
    assert not tf["eroded"][7, 3], "the 1-cell bridge must be cut by over-etch"
    assert not tf["eroded"][13, :].any(), "the 1-cell spur must vanish"
    assert tf["dilated"][7, :8].all(), "under-etch must fuse the two resonators"
    assert tf["eroded"][2:7, 0].any(), \
        "three_fields must route the trace-rooted convention to the erosion"
    assert not tf["dilated"][:, 0].all(), \
        "three_fields' dilated field must not fill the trace-adjacent row"
    print("  -> over-etch CUTS the 1-cell bridge and DELETES the 1-cell spur; "
          "under-etch FUSES the two resonators. One mask, both failure modes.")
    print(f"  -> trace-adjacent row j=0: rooted metal SURVIVES the over-etch "
          f"({int(tf['eroded'][:, 0].sum())} cells) and the under-etch does "
          f"NOT fill the row ({int(tf['dilated'][:, 0].sum())} of "
          f"{mixed.shape[0]}).")

    # ---- (f) robust_score + tolerance_report on synthetic responses ------
    print("\n[f] robust_score / tolerance_report")
    print("    SYNTHETIC responses from an order-N bandstop model. These are "
          "NOT solves and stand")
    print("    for no arm -- they exist to exercise the reporting path with "
          "traces of the right shape.")
    g = scoring_grid_mhz()

    # Design R -- COARSE features only (a stub-like layout, every feature many
    # cells wide). The etch cannot change its topology, so all three fields
    # carry the same two resonators and the only effect is the +-0.75 % centre
    # drift computed in [c]: over-etch shortens (f up), under-etch lengthens
    # (f down).
    coarse_fields = {
        "eroded":  [(5289, 300, 26, 5), (5823, 200, 26, 5)],
        "nominal": [(5250, 300, 26, 5), (5775, 200, 26, 5)],
        "dilated": [(5211, 300, 26, 5), (5727, 200, 26, 5)],
    }
    # Design F -- carries 1-CELL features, like the ``mixed`` mask in [e].
    # Nominally it is the better design. Over-etch CUTS the 1-cell bridge, so
    # the upper resonator decouples: its notch collapses to ~6 dB and runs off
    # to 5.95 GHz. Under-etch CLOSES the 1-cell gap, fusing the pair into one
    # wide merged notch that blocks the inter-band gap.
    fine_fields = {
        "eroded":  [(5250, 290, 28, 8), (5950, 180, 6, 3)],
        "nominal": [(5250, 300, 30, 8), (5775, 200, 30, 8)],
        "dilated": [(5500, 780, 26, 6)],
    }

    reports = {}
    for name, spec in (("R -- coarse features only (synthetic)", coarse_fields),
                       ("F -- carries 1-cell features (synthetic)", fine_fields)):
        responses = {k: {"freqs_mhz": g, "il_db": _il_bandstop(g, z, 0.1)}
                     for k, z in spec.items()}
        rr = robust_score(responses, etch=cal)
        reports[name] = rr
        print()
        print(tolerance_report(rr, design=name))

    a = reports["R -- coarse features only (synthetic)"]
    b = reports["F -- carries 1-cell features (synthetic)"]
    assert b.M_nominal < a.M_nominal, \
        "fixture: the fine-feature design must look BETTER nominally"
    assert b.M_worst > a.M_worst, \
        "fixture: and WORSE once the etch window is applied"
    assert b.M_spread > a.M_spread
    assert a.M_worst == max(r.M for r in a.per_field.values())
    assert a.Omega_worst == min(r.Omega for r in a.per_field.values())
    assert b.term_deltas["eroded"]["S_U"] > 10, "severed resonator -> S_U"
    assert b.term_deltas["dilated"]["S_G"] > 10, "fused pair -> S_G"
    assert abs(b.notch_shift_mhz["eroded"]["U"]) >= 100, \
        "the severed upper notch must show a large free-running drift"
    assert max(abs(v) for v in a.notch_shift_mhz["dilated"].values()) <= 100, \
        "the coarse design must drift only by the etch-induced length change"
    print("\n[f] ranking (robust_rank_key -- worst case first, the PI's margin "
          "framing):")
    for nm, rr in sorted(reports.items(), key=lambda kv: robust_rank_key(kv[1])):
        print(f"    M_worst={rr.M_worst:6.2f}  M_nom={rr.M_nominal:6.2f}  "
              f"spread={rr.M_spread:5.2f}  fragility={rr.fragility_db:+6.2f}  {nm}")
    print("    Ranking on NOMINAL M would put F first (0.01 vs 0.05). Ranking "
          "on WORST-CASE M reverses")
    print("    it. That reversal, and the fact that F's failure is a TOPOLOGY "
          "collapse rather than a")
    print("    frequency drift, is what this module exists to expose.")

    # ---- (g) guard rails -------------------------------------------------
    print("\n[g] guard rails")
    for bad, why in (
        ({"nominal": (g, np.zeros(len(g)))}, "missing fields"),
        ({**{k: (g, np.zeros(len(g))) for k in FIELD_ORDER},
          "extra": (g, np.zeros(len(g)))}, "unexpected field"),
    ):
        try:
            robust_score(bad)
        except ValueError as exc:
            print(f"  ok  {why}: {exc}")
        else:                                          # pragma: no cover
            raise AssertionError(f"expected ValueError for {why}")
    try:
        erode(np.zeros((3, 3), dtype=bool), -1, outside=0)
    except ValueError as exc:
        print(f"  ok  negative offset: {exc}")
    else:                                              # pragma: no cover
        raise AssertionError("expected ValueError for negative cells")
    try:
        erode(np.zeros((3, 3, 3), dtype=bool), 1, outside=0)
    except ValueError as exc:
        print(f"  ok  non-2-D mask: {exc}")
    else:                                              # pragma: no cover
        raise AssertionError("expected ValueError for 3-D mask")

    # -- minor (a): the boundary convention has NO default -----------------
    small = np.zeros((5, 5), dtype=bool)
    small[1:4, 1:4] = True
    for fn, nm in ((erode, "erode"), (three_fields, "three_fields")):
        try:
            fn(small, 1)
        except TypeError as exc:
            print(f"  ok  {nm}() refuses to guess a boundary convention: {exc}")
        else:                                          # pragma: no cover
            raise AssertionError(f"{nm}() must REQUIRE outside=")
    try:
        dilate(small, 1, outside=OUTSIDE_TRACE_AT_Y_LO)
    except TypeError as exc:
        print(f"  ok  dilate() takes no boundary convention at all: {exc}")
    else:                                              # pragma: no cover
        raise AssertionError("dilate() must not accept outside=")

    # -- minor (b): per-field overrides of the scoring problem are refused --
    for k, v in (("band_l", (5150, 5350)), ("band_u", (5725, 5825)),
                 ("guard_mhz", 100), ("f_lo", 3100), ("f_hi", 8600)):
        resp_bad = {n: {"freqs_mhz": g,
                        "il_db": _il_bandstop(g, coarse_fields[n], 0.1)}
                    for n in FIELD_ORDER}
        resp_bad["dilated"][k] = v
        try:
            robust_score(resp_bad)
        except ValueError as exc:
            assert k in str(exc), exc
        else:                                          # pragma: no cover
            raise AssertionError(f"per-field {k!r} override must be refused")
    print("  ok  per-field band_l/band_u/guard_mhz/f_lo/f_hi overrides refused "
          "(they would desynchronize score() from _search_windows)")
    # unquotable calibration must annotate the report rather than fail silently
    coarse = calibrate_etch(cell_um=CELL_COARSE_UM, cells=1)
    rr_c = robust_score({k: {"freqs_mhz": g, "il_db": _il_bandstop(g, z, 0.1)}
                         for k, z in coarse_fields.items()}, etch=coarse)
    assert rr_c.notes and "not quotable" in rr_c.notes[0]
    print(f"  ok  coarse-mesh calibration annotated: {rr_c.notes[0]}")
    # validity must PROPAGATE: an unsettled field poisons the worst case.
    # (score() stores validity via asdict, which drops the .ok property --
    # this is the regression test for that.)
    good = Validity(settled=True, settling_worst_db=-102.8, passivity_worst=0.01)
    bad = Validity(settled=False, settling_worst_db=-18.8, passivity_worst=0.01)
    resp = {k: {"freqs_mhz": g, "il_db": _il_bandstop(g, z, 0.1),
                "validity": good} for k, z in coarse_fields.items()}
    assert robust_score(resp, etch=cal).all_valid is True
    resp["eroded"] = dict(resp["eroded"], validity=bad)
    rr_v = robust_score(resp, etch=cal)
    assert rr_v.all_valid is False and any("eroded" in n for n in rr_v.notes), \
        "an unsettled field must be named and must make the number unquotable"
    print(f"  ok  validity propagates: {rr_v.notes[-1]}")
    assert robust_score({k: {"freqs_mhz": g, "il_db": _il_bandstop(g, z, 0.1)}
                         for k, z in coarse_fields.items()}).all_valid is None
    print("  ok  validity absent -> all_valid is None (unknown, not True)")

    # the RELAXED thresholds (arm-D kill gate) route through unchanged
    rr_r = robust_score({k: {"freqs_mhz": g, "il_db": _il_bandstop(g, z, 0.1)}
                         for k, z in coarse_fields.items()},
                        thr=RELAXED, etch=cal)
    assert rr_r.M_worst <= rr_c.M_worst
    print(f"  ok  RELAXED thresholds pass through: M_worst "
          f"{rr_c.M_worst:.2f} (SCORE) -> {rr_r.M_worst:.2f} (RELAXED)")

    print("\n" + "=" * 96)
    print("ALL SELF-TESTS PASSED")
    print("=" * 96)
    return 0


if __name__ == "__main__":
    raise SystemExit(_selftest())
