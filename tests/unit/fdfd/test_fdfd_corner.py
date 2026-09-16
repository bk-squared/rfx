"""Validation gates for the D2c study
``validation/fdfd/corner_convergence.py``: ONE right-angle corner on the very
same 3-D FDFD fixture the spiral and the straight bar use
(``rfx.fdfd.spiral``, ``dut_kind="bend"``, an additive hook), against the
area-exact Greenhouse referee, to test whether corners explain what D2b's
corner-free bar left of the spiral study's residual.

The study runs two shapes x three levels x a ``pad_cells`` wall ladder
(34.3 min of solves, up to 18.8 GB and 466 s for its largest single solve).
This file re-runs the COARSEST level of both shapes live (``base_dx = W``,
N = 22044 and 17344) and the AD check on a cheap 40 um bend (N = 11520), and
asserts the parts of the protocol those grids support plus everything that
needs no solver at all (the hook's geometry, the short standard's cell
ranges, the referee, the corner convention, the corner count). The finer
levels, the wall ladders and the attribution are in the study's JSON; T8
gates that JSON against the fresh W/1 solves so the recorded numbers cannot
drift silently, and T9 / T10 re-derive gates K3 and K5 from it with the
study's own code.

Every number below was measured in this session; the wall clock was shared
with another solver job, so the timings are upper bounds.

Gates here
----------
T1  the ``dut_kind="bend"`` hook builds exactly the documented fixture --
    ``theta = (l1, l2, W)`` (a THREE-tuple, unlike the bar's two-tuple),
    four x breakpoints at ``+-(l2 +- W)/2`` and four y breakpoints at
    ``+-(l1 +- W)/2``, all on the nominal grid; the drawn L polygon's area
    is ``(l1 + l2 + W) W`` and each footprint square's is ``W^2``, both to
    4.5e-16 relative; both lead columns on M2 with the inner one LEFT of
    the outer one and, unlike the bar's and the spiral's, on a DIFFERENT
    row; no M1 / via metal in the DUT or the OPEN; the three fixtures in
    the expected containment order. Build only, no solve.
T2  the short standard is the one the study's referee subtracts: the post
    spans ``i in [inner.i1 + 1, outer.i0 - 1)``, the UNION of the two
    terminal rows in ``j`` and the ground up to the top of M2 in ``z``;
    each arm sits on its OWN column's row and reaches the post; both arms
    are x-directed and ``W/2 + one cell`` = 15.000 um long at
    ``base_dx = W`` (the value ``arm_lengths`` reports and the referee
    uses). Build only.
T3  the hook is ADDITIVE: with ``dut_kind`` left at ``"spiral"`` the D2
    fixture still builds to 21 x 22 x 15 cells, N = 23062, a three-tuple
    ``theta``, terminals on DIFFERENT metal levels and M2 + M1 + via
    polygons; the bar still builds its two ``W x W`` end squares on M2. For
    both, the generalised short standard still puts BOTH arms and the post
    on the single shared terminal row -- the property the bend's
    two-row generalisation had to preserve. Build only.
T4  the corner convention: the area-exact rectangles sum to
    ``(l1 + l2) W`` and their UNION equals that sum, both to 2.3e-16
    relative, while the delivered Greenhouse convention misses the union by
    -1.250 % (one quadrant of the corner square counted twice, the opposite
    one not at all). Host numpy.
T5  the referee is physical and the corner excess is NEGATIVE with the
    documented mechanism: the bend's two perpendicular bars have EXACTLY
    zero mutual inductance, the equal-length straight bar's self term is
    26.989 pH larger and its ground image 19.211 pH more negative, giving
    a strip-only excess of -7.778 pH = -7.10 % of the bar. Host numpy.
T6  gate K3 at the coarsest level, live: the de-embedded ``L_diff`` of the
    bend (68.5481 pH) and of the bar (75.2438 pH) at ``pad_cells = 4``
    reproduce the study's JSON to 1e-7 relative, and their difference has
    the same sign as the referee's -- -6.6957 pH against -7.7934 pH, i.e.
    the FDFD reproduces 85.9 % of the corner excess at one cell across W
    on the raw (wall-biased) pad-4 grids while its absolute L is 26.2 %
    and 25.3 % low.
T7  gate K2 on the hook's derivatives, live on a cheap 40 um x 40 um bend
    (N = 11520, 9.8 s for the gradient and 39.4 s for the eight FD
    solves): ``jax.grad`` of the de-embedded L in ``theta = (l1, l2, W)``
    against FD4 with 1 % steps (which move L by ~1e-2 relative, far above
    the ~1e-9 LU noise floor) agrees to 3.07e-06 (``dL/dl1``) and 9.32e-07
    (``dL/dW``), gate 1e-4.
T8  the study's JSON exists, is self-consistent, and its conclusion holds:
    *  THE THREE-POINT RULE (K1). W/1 and W/2 carry a full ``pad_cells``
       4/6/8 ladder on BOTH legs and are admitted; W/3 carries [0, 2]
       because three of its six grids are above ``N_MAX = 110000``
       (bend pad 6 at 141440 and pad 8 at 182784, bar pad 8 at 129684, all
       recorded in ``study["skipped"]``); with no ladder at all on its bend
       leg it has no wall-corrected number, only its raw pad-4 ones.
    *  the wall bias depends on the SHAPE: +0.794 % (bend) against
       +1.475 % (bar) at W/1 and +1.763 % against +3.702 % at W/2, which is
       why the raw pad-4 discrepancy moves the WRONG way with refinement
       (1.0977 -> 1.9464 -> 3.2219 pH) while the wall-corrected one falls
       (0.5323 +- 0.4151 -> 0.1313 +- 1.0251 pH).
    *  the wall-corrected corner excess is 0.9317 and 0.9831 of the
       referee's at one and two cells across W, where the absolute L is
       -25.62 % and -13.42 %.
    *  the corner COUNT of the D2 spiral is 8, counted from
       ``gds.rect_spiral``'s centreline (7 joints between the 8 sides, 1
       where the last side meets the inner extension, 0 for the collinear
       lead and 0 for the underpass via).
    *  the attribution (K5) FAILS, in ONE sign convention, FDFD minus
       referee: 8 x (+0.1313 +- 1.0251) pH = [-7.1506, +9.2517] pH misses
       the invariant ladder's continuum-limit residual [-15.1813, -8.2489]
       pH (``invariant_ladder.json``, the primary comparison) by 1.0983 pH,
       the same limit after its corrections, [-15.4739, -8.4415] pH, by
       1.2908 pH, and the superseded D2-minus-D2b remainder
       [-19.1913, -8.4094] pH by 1.2588 pH; the corner term's central
       value, +1.0505 pH, has the opposite sign to all three.
T9  gate K3 is evaluated on the ADMITTED levels only -- W/1 and W/2,
    -7.2611 and -7.6533 pH against the referee's -7.7934 and -7.7846 pH,
    the set K4 and K5 use. W/3's raw pad-4 excess (-4.5607 pH against
    -7.7826 pH, ratio 0.5860) is reported under ``excluded_informational``
    and never gated. Re-evaluated from the JSON with the study's own
    ``evaluate_gates``: the recorded gate is reproduced exactly, a W/3 of
    the wrong sign leaves K3 passing and an admitted level of the wrong
    sign fails it. No solver.
T10 gate K5 re-derived from the recorded excess and corner count with the
    study's own ``attribution``, which reads ``invariant_ladder.json``
    itself, reproduces the recorded block exactly; its primary residual is
    that JSON's ``residual.limit_rel_range`` x ``referee.total`` (-4.5582 %
    to -2.4767 % of 333.055 pH), and an invariant ladder whose referee is
    not D2's is refused. No solver.

x64 is scoped per test through ``tests._x64_compat.enable_x64`` (never
flipped at module level); the static models and the coarse solves are cached
for the module. Measured runtime of this file: 69 s.
"""
from __future__ import annotations

import copy
import importlib.util
import json
import pathlib
import time

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx.fdfd import gds
from rfx.fdfd import spiral as sm
from tests._x64_compat import enable_x64

pytest.importorskip("shapely")

REPO = pathlib.Path(__file__).resolve().parents[3]
STUDY_PATH = REPO / "validation" / "fdfd" / "corner_convergence.py"
JSON_PATH = REPO / "validation" / "fdfd" / "corner_convergence.json"
INVARIANT_PATH = REPO / "validation" / "fdfd" / "invariant_ladder.json"

COARSE = 1.0              # base_dx = W: bend N = 22044, bar N = 17344
L_CHEAP = 40e-6           # the cheap bend used for the live AD check (N = 11520)

_CACHE: dict = {}


def _load(path: pathlib.Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _study():
    if "study" not in _CACHE:
        _CACHE["study"] = _load(STUDY_PATH, "corner_convergence_study")
    return _CACHE["study"]


def _referee():
    if "referee" not in _CACHE:
        _CACHE["referee"] = _study().load_referee()
    return _CACHE["referee"]


def _model(kind: str, div: float = COARSE):
    k = f"model:{kind}:{div}"
    if k not in _CACHE:
        _CACHE[k] = _study().build(kind, div)
    return _CACHE[k]


def _solve(kind: str, div: float = COARSE):
    """Cached three-fixture volumetric solve of one study fixture."""
    st = _study()
    k = f"solve:{kind}:{div}"
    if k not in _CACHE:
        t0 = time.time()
        val = st.l_dut(_model(kind, div))
        jax.block_until_ready(val)
        _CACHE[k] = {"L": float(np.real(val)), "seconds": time.time() - t0}
    return _CACHE[k]


def _json() -> dict:
    if "json" not in _CACHE:
        assert JSON_PATH.exists(), f"run {STUDY_PATH} first"
        _CACHE["json"] = json.loads(JSON_PATH.read_text())
    return _CACHE["json"]


# ----------------------------------------------------------------------------
# T1 / T2 / T3: the hook (build only, no solver)

def test_bend_dut_kind_builds_the_documented_fixture():
    """T1. ``SpiralSpec(dut_kind="bend", bend_l1=l1, bend_l2=l2)`` must give
    exactly the fixture the study's referee assumes, and must not quietly
    reuse the spiral's underpass / via topology or the bar's collinear
    columns."""
    with enable_x64():
        st = _study()
        m = _model("bend")
        l1, l2, w = st.L1, st.L2, st.WIDTH
        assert m.spec.dut_kind == "bend"
        assert m.theta_nominal == (l1, l2, w), m.theta_nominal
        assert len(m.theta_nominal) == 3                       # NOT the bar's two-tuple

        bx, by = sm.bend_edge_coordinates(jnp.asarray(m.theta_nominal))
        want_x = np.array([-(l2 + w), -(l2 - w), l2 - w, l2 + w]) / 2.0
        want_y = np.array([-(l1 + w), -(l1 - w), l1 - w, l1 + w]) / 2.0
        assert np.allclose(np.asarray(bx), want_x, rtol=1e-15, atol=0.0), np.asarray(bx)
        assert np.allclose(np.asarray(by), want_y, rtol=1e-15, atol=0.0), np.asarray(by)
        for v in want_x:
            assert np.min(np.abs(m.x_nom - v)) <= 1e-15 * l2
        for v in want_y:
            assert np.min(np.abs(m.y_nom - v)) <= 1e-15 * l1

        # the drawn geometry: one mitred L polygon of area (l1 + l2 + W) W and
        # two W x W footprint squares, all on M2 only
        assert set(m.spiral.polygons) == {sm.KEY_M2}, "the bend has no underpass and no via"
        polys = m.spiral.polygons[sm.KEY_M2]
        assert len(polys) == 3, len(polys)
        assert abs(abs(gds.polygon_area(polys[0])) / ((l1 + l2 + w) * w) - 1.0) <= 5e-16
        for q in polys[1:]:
            assert abs(abs(gds.polygon_area(q)) / (w * w) - 1.0) <= 5e-16
        assert abs(m.spiral.length - (l1 + l2 + w)) <= 1e-15 * l1

        # cells: the strip is on M2, nothing on M1 / via in the DUT or the OPEN
        sk = m.spec.stack
        zc = 0.5 * (m.z[:-1] + m.z[1:])
        k_m2 = (zc > sk.z_m2) & (zc < sk.z_m2 + sk.t_m2)
        k_low = zc < sk.z_m2
        dut, opn, sho = (m.cells[k] for k in ("dut", "open", "short"))
        assert dut[:, :, k_m2].sum() > 0
        assert not opn[:, :, k_m2].any(), "the OPEN fixture must be the columns alone"
        assert np.all(dut | opn == dut), "OPEN must be a subset of the DUT"
        assert np.all(sho | opn == sho), "OPEN must be a subset of the SHORT"
        # the only DUT metal below M2 is the two lead columns
        assert np.array_equal(dut[:, :, k_low], opn[:, :, k_low])

        # both columns on M2, inner LEFT of outer and -- the bend's own property --
        # on a DIFFERENT row
        outer, inner = m.columns
        assert (outer.k0, outer.k1) == (inner.k0, inner.k1), (outer, inner)
        assert inner.i1 < outer.i0, "the second column must be left of the first"
        assert (inner.j0, inner.j1) != (outer.j0, outer.j1), "the bend's columns share no row"
        assert inner.j0 > outer.j1, (outer, inner)
        for c, xc, yc in ((outer, +l2 / 2, -l1 / 2), (inner, -l2 / 2, +l1 / 2)):
            assert abs(m.x_nom[c.i0] - (xc - w / 2)) <= 1e-15 * l2
            assert abs(m.x_nom[c.i1] - (xc + w / 2)) <= 1e-15 * l2
            assert abs(m.y_nom[c.j0] - (yc - w / 2)) <= 1e-15 * l1
            assert abs(m.y_nom[c.j1] - (yc + w / 2)) <= 1e-15 * l1

        sm.check_feasible(m, (l1, l2, w))
        with pytest.raises(ValueError):
            sm.check_feasible(m, (0.5 * w, l2, w))             # a leg shorter than the width
        gaps = np.asarray(sm.breakpoint_gaps(m, jnp.asarray((l1, l2, w))))
        assert np.all(gaps > 0), gaps
        with pytest.raises(ValueError):
            sm.bend_geometry(0.5 * w, l2, w)


def test_bend_short_standard_is_the_bridge_the_referee_subtracts():
    """T2. The short standard's cells must be exactly what the study's
    ``bend_bridge_segments`` values: an x-directed arm on EACH column's own
    terminal row running to a common post one cell inside each column, the
    post spanning the union of the two rows from the ground up through M2."""
    with enable_x64():
        st = _study()
        m = _model("bend")
        outer, inner = m.columns
        sk = m.spec.stack
        zc = 0.5 * (m.z[:-1] + m.z[1:])
        k_m2 = np.nonzero((zc > sk.z_m2) & (zc < sk.z_m2 + sk.t_m2))[0]
        short_only = m.cells["short"] & ~m.cells["open"]

        post_i0, post_i1 = inner.i1 + 1, outer.i0 - 1
        assert post_i1 > post_i0, (post_i0, post_i1)
        # the post: the only short-standard metal that reaches the ground cell
        ground_row = short_only[:, :, 0]
        ii, jj = np.nonzero(ground_row)
        assert ii.min() == post_i0 and ii.max() == post_i1 - 1, (ii.min(), ii.max())
        assert jj.min() == min(outer.j0, inner.j0), jj.min()
        assert jj.max() == max(outer.j1, inner.j1) - 1, jj.max()
        # ... and it stops at the top of M2
        assert short_only[post_i0, outer.j0, k_m2[-1]]
        assert not short_only[post_i0, outer.j0, k_m2[-1] + 1]

        # each arm on its own row, reaching from the post to its column
        for c, i_lo, i_hi in ((outer, post_i0, outer.i1), (inner, inner.i0, post_i1)):
            row = short_only[:, c.j0:c.j1, k_m2[0]]
            assert row.shape[1] == c.j1 - c.j0
            on = np.nonzero(row.any(axis=1))[0]
            assert on.min() == i_lo and on.max() == i_hi - 1, (on.min(), on.max(), i_lo, i_hi)
            # the arm is one row thick: nothing on the neighbouring rows except
            # where the post is
            assert not short_only[i_hi - 1, c.j1 % m.shape[1], k_m2[0]] or i_hi - 1 < post_i1

        # the arm lengths the referee uses: W/2 + one cell, both x-directed
        arm_o, arm_i = st.arm_lengths(m)
        cell = float(m.x_nom[outer.i0] - m.x_nom[outer.i0 - 1])
        assert abs(arm_o - (0.5 * st.WIDTH + cell)) <= 1e-15 * st.L2
        assert abs(arm_i - arm_o) <= 1e-15 * st.L2
        assert abs(arm_o - 15.0e-6) <= 1e-12, arm_o          # measured at base_dx = W
        sg = _referee()
        segs = st.bend_bridge_segments(sg, arm_o, arm_i)
        assert len(segs) == 2
        for s in segs:
            assert sg.segment_axis(s) == 0, "both bridge arms are x-directed"
            d = np.asarray(s.end, float) - np.asarray(s.start, float)
            assert d[0] < 0, "both arms carry the differential current in -x"
            assert abs(abs(d[0]) - arm_o) <= 1e-15 * st.L2


def test_spiral_and_bar_fixtures_are_unchanged_by_the_bend_hook():
    """T3 (regression). The bend hook generalises the short standard's single
    terminal row to one row per column; with ``dut_kind`` left at
    ``"spiral"`` or set to ``"bar"`` the two rows coincide and the fixture
    must be the one those studies recorded. The specs are written out here
    rather than imported, so this test does not move when their drivers do."""
    with enable_x64():
        st = _study()
        spec = sm.SpiralSpec(n_turns=2, r_out=52e-6, spacing=10e-6, width=10e-6, lead=15e-6,
                             base_dx=10e-6, margin=st.MARGIN, stack=st.stack(),
                             port_gap_cells=1, pad_cells=st.PAD_CELLS, pad_ratio=st.PAD_RATIO)
        assert spec.dut_kind == "spiral"
        assert spec.theta == (52e-6, 10e-6, 10e-6), spec.theta
        m = sm.build_spiral(spec)
        assert len(m.theta_nominal) == 3, m.theta_nominal
        assert tuple(m.shape) == (21, 22, 15), m.shape
        assert m.n_unknowns == 23062, m.n_unknowns
        c0, c1 = m.columns
        assert (c0.k0, c0.k1) != (c1.k0, c1.k1), (c0, c1)     # M2 and M1 terminals
        assert set(m.spiral.polygons) == {sm.KEY_M2, sm.KEY_M1, sm.KEY_VIA}

        bar = _model("bar")
        assert set(bar.spiral.polygons) == {sm.KEY_M2}
        assert bar.theta_nominal == (st.L1 + st.L2, st.WIDTH), bar.theta_nominal

        # the short standard of BOTH still lives on the single shared row
        for mm in (m, bar):
            a, b = mm.columns
            assert (a.j0, a.j1) == (b.j0, b.j1), (a, b)
            short_only = mm.cells["short"] & ~mm.cells["open"]
            jj = np.nonzero(short_only.any(axis=(0, 2)))[0]
            assert jj.min() == a.j0 and jj.max() == a.j1 - 1, (jj, a)


# ----------------------------------------------------------------------------
# T4 / T5: the referee and the corner convention (host numpy, no solver)

def test_area_exact_corner_convention_covers_the_corner_square_once():
    """T4. Gate K_area: the area-exact convention's two rectangles sum to
    ``(l1 + l2) W`` AND are disjoint; the delivered Greenhouse convention
    gives the same sum but a union 1.25 % smaller, which is the
    double-counted quadrant."""
    with enable_x64():
        st = _study()
        gate = st.area_gate(_referee())
        assert gate["passed"], gate
        ae, gh = gate["area_exact"], gate["greenhouse"]
        assert abs(ae["sum_over_expected_minus_1"]) <= 2.3e-16, ae
        assert abs(ae["union_over_expected_minus_1"]) <= 2.3e-16, ae
        assert abs(ae["union_over_sum_minus_1"]) <= 2.3e-16, ae
        assert abs(gh["sum_over_expected_minus_1"]) <= 1e-15, gh
        assert abs(gh["union_over_sum_minus_1"] + 0.0125) <= 1e-9, gh


def test_referee_corner_excess_is_negative_with_the_documented_mechanism():
    """T5. At equal total centreline length the bend must be the LOWER
    inductance, and for the stated reason: its two halves are perpendicular
    (mutual exactly zero) while the straight bar's self term contains the
    collinear mutual of its halves; the ground images give part of it
    back."""
    with enable_x64():
        st, sg = _study(), _referee()
        bend = sg.greenhouse_terms(st.bend_segments(sg), ground_height=st.GROUND_H)
        bar = sg.greenhouse_terms(st.bar_segments(sg, st.L1 + st.L2), ground_height=st.GROUND_H)
        assert bend.self_sum > 0 and bar.self_sum > 0
        assert bend.image_sum < 0 and bar.image_sum < 0
        assert bend.mutual_sum == 0.0, "perpendicular bars must not couple"
        assert bar.mutual_sum == 0.0, "one segment has no mutual partner"
        assert abs(bend.total - 101.8310e-12) <= 5e-16
        assert abs(bar.total - 109.6093e-12) <= 5e-16
        assert abs((bend.self_sum - bar.self_sum) + 26.989e-12) <= 1e-15
        assert abs((bend.image_sum - bar.image_sum) - 19.211e-12) <= 1e-15
        excess = bend.total - bar.total
        assert excess < 0.0, excess
        assert abs(excess + 7.778e-12) <= 1e-15, excess
        assert abs(excess / bar.total + 0.0710) <= 1e-4


# ----------------------------------------------------------------------------
# T6 / T7: live solves at the coarsest level

def test_corner_excess_sign_at_the_coarsest_level():
    """T6. Gate K3 at ``base_dx = W``, live: both de-embedded values
    reproduce the study's JSON and the corner excess has the referee's sign
    and 86 % of its size, while the absolute L is 25-26 % low."""
    with enable_x64():
        st, sg = _study(), _referee()
        bend, bar = _solve("bend"), _solve("bar")
        rec = _json()["levels"]
        assert abs(bend["L"] / rec["bend"]["1"]["4"]["L_dut"] - 1.0) <= 1e-7
        assert abs(bar["L"] / rec["bar"]["1"]["4"]["L_dut"] - 1.0) <= 1e-7
        assert abs(bend["L"] - 68.5481e-12) <= 5e-16, bend["L"]
        assert abs(bar["L"] - 75.2438e-12) <= 5e-16, bar["L"]

        arm_o, arm_i = st.arm_lengths(_model("bend"))
        ref_b = st.referee_point(sg, "bend", arm_o, arm_i)
        ref_r = st.referee_point(sg, "bar", *st.arm_lengths(_model("bar")))
        ex_f = bend["L"] - bar["L"]
        ex_r = ref_b["L_deembedded"] - ref_r["L_deembedded"]
        assert ex_f < 0.0 and ex_r < 0.0, (ex_f, ex_r)
        assert np.sign(ex_f) == np.sign(ex_r)
        assert abs(ex_f + 6.6957e-12) <= 5e-16, ex_f
        assert abs(ex_r + 7.7934e-12) <= 5e-16, ex_r
        assert abs(ex_f / ex_r - 0.8591) <= 1e-3, ex_f / ex_r
        # the absolute L is 25-26 % low at this level -- stated as measured
        assert abs(bend["L"] / ref_b["L_deembedded"] - 1.0 + 0.2621) <= 1e-3
        assert abs(bar["L"] / ref_r["L_deembedded"] - 1.0 + 0.2527) <= 1e-3


def test_ad_vs_fd4_on_the_bend_hook():
    """T7. Gate K2's claim, on a cheap 40 um x 40 um bend: ``jax.grad`` of
    the de-embedded ``L_diff`` in ``theta = (l1, l2, W)`` against FD4 with
    1 % steps. The FD steps move L by ~1e-2 relative, far above the ~1e-9 LU
    noise floor, so what is measured is the FD truncation."""
    with enable_x64():
        st = _study()
        m = st.build("bend", 1.0, st.PAD_CELLS, l1=L_CHEAP, l2=L_CHEAP)
        assert m.n_unknowns == 11520, m.n_unknowns
        theta0 = (L_CHEAP, L_CHEAP, st.WIDTH)

        def f(t):
            return jnp.real(st.l_dut(m, t))

        t0 = time.time()
        val, grad = jax.value_and_grad(f)(jnp.asarray(theta0, dtype=jnp.float64))
        jax.block_until_ready(grad)
        assert abs(float(val) - 14.3798e-12) <= 5e-16, float(val)
        fd = []
        for k in (0, 2):
            def g(v: float, k: int = k) -> float:
                t = list(theta0)
                t[k] = v
                return float(f(jnp.asarray(t, dtype=jnp.float64)))
            fd.append(st.fd4(g, theta0[k], st.FD_STEP_REL * theta0[k]))
        rel = [abs(float(grad[k]) - v) / abs(v) for k, v in zip((0, 2), fd)]
        print(f"  AD/FD4 on the bend hook: dL/dl1 {float(grad[0]):.6e} vs {fd[0]:.6e}, "
              f"dL/dW {float(grad[2]):.6e} vs {fd[1]:.6e}; rel {rel} "
              f"({time.time() - t0:.0f} s)")
        assert max(rel) <= 1e-4, rel
        assert rel[0] <= 1e-5 and rel[1] <= 1e-5, rel     # measured 3.07e-6 and 9.32e-7


# ----------------------------------------------------------------------------
# T8: the study's JSON

def test_study_json_holds_the_conclusion():
    """T8. The recorded study is self-consistent and says what the module
    docstring says -- including the two facts that keep "K5 passed" from
    being read as "the corners explain it"."""
    d = _json()
    g = d["gates"]
    exc = d["excess"]["levels"]

    # --- K1: the three-point rule, structurally ---------------------------
    assert g["K1"]["passed"]
    assert g["K1"]["admitted_levels"] == ["W/1", "W/2"], g["K1"]["admitted_levels"]
    assert g["K1"]["excluded_levels"] == ["W/3"], g["K1"]["excluded_levels"]
    assert g["K1"]["n_max"] == 110_000
    for dk in ("1", "2"):
        assert exc[dk]["admitted"] is True
        assert min(exc[dk]["wall_ladder_points"]) >= 3, exc[dk]["wall_ladder_points"]
        assert "discrepancy" in exc[dk] and "uncertainty" in exc[dk]
        assert "discrepancy_two_point_EXCLUDED" not in exc[dk]
    assert exc["3"]["admitted"] is False
    assert exc["3"]["wall_ladder_points"] == [0, 2], exc["3"]["wall_ladder_points"]
    assert "discrepancy" not in exc["3"]
    assert set(d["skipped"]) == {"bend_W3_pad6", "bend_W3_pad8", "bar_W3_pad8"}, d["skipped"]
    for k, n in (("bend_W3_pad6", 141440), ("bend_W3_pad8", 182784), ("bar_W3_pad8", 129684)):
        assert d["skipped"][k]["n_unknowns"] == n, (k, d["skipped"][k])
        assert d["skipped"][k]["n_unknowns"] > 110_000

    # --- K2 ----------------------------------------------------------------
    assert g["K2"]["passed"] and g["K2"]["worst"] <= 1e-4
    assert g["K2"]["worst"] <= 2e-7, g["K2"]["worst"]          # measured 1.59e-07
    assert g["K2"]["fd_step_rel"] == 0.01
    assert g["K2"]["n_unknowns"] == 22044

    # --- K3: sign agreement on the ADMITTED levels, and the ratios ---------
    assert g["K3"]["passed"] is True
    per = g["K3"]["per_level"]
    assert set(per) == {"W/1", "W/2"}, set(per)               # W/3 is not gated (T9)
    for v in per.values():
        assert v["sign_fdfd"] == -1 and v["sign_referee"] == -1, v
    assert abs(per["W/1"]["ratio"] - 0.9317) <= 1e-3, per["W/1"]["ratio"]
    assert abs(per["W/2"]["ratio"] - 0.9831) <= 1e-3, per["W/2"]["ratio"]
    # the corner excess is reproduced far better than the absolute L
    for dk, gap in (("1", 0.2562), ("2", 0.1342)):
        assert abs(exc[dk]["gap_wall_corrected"]["bend"] + gap) <= 1e-3, exc[dk]

    # --- K4: the wall-corrected discrepancy falls, the raw one does not ----
    assert g["K4"]["passed"]
    adm = g["K4"]["admitted"]
    assert len(adm) == 2
    assert abs(adm[0]["discrepancy"] - 0.5323e-12) <= 5e-16, adm[0]
    assert abs(adm[0]["uncertainty"] - 0.4151e-12) <= 5e-16, adm[0]
    assert abs(adm[1]["discrepancy"] - 0.1313e-12) <= 5e-16, adm[1]
    assert abs(adm[1]["uncertainty"] - 1.0251e-12) <= 5e-16, adm[1]
    assert abs(adm[1]["discrepancy"]) < abs(adm[0]["discrepancy"])
    raw = [r["discrepancy"] for r in g["K4"]["raw_pad4"]]
    assert g["K4"]["raw_pad4_decreases"] is False
    assert raw[0] < raw[1] < raw[2], raw                        # 1.0977 < 1.9464 < 3.2219 pH
    assert abs(raw[2] - 3.2219e-12) <= 5e-16, raw

    # the wall bias depends on the shape, which is why that happens
    wl = d["wall_ladder"]
    assert abs(wl["bend"]["1"]["rel_pad4"] - 0.00794) <= 1e-5
    assert abs(wl["bar"]["1"]["rel_pad4"] - 0.01475) <= 1e-5
    assert abs(wl["bend"]["2"]["rel_pad4"] - 0.01763) <= 1e-5
    assert abs(wl["bar"]["2"]["rel_pad4"] - 0.03702) <= 1e-5
    assert wl["bar"]["1"]["rel_pad4"] > 1.8 * wl["bend"]["1"]["rel_pad4"]
    assert wl["bar"]["2"]["rel_pad4"] > 2.0 * wl["bend"]["2"]["rel_pad4"]

    # --- the corner count --------------------------------------------------
    cc = d["corner_count"]
    assert cc["n_corners"] == 8, cc
    assert cc["n_right_angle"] == 8 and cc["n_collinear"] == 1 and cc["n_reversal"] == 0
    assert cc["turns_times_four"] == 8
    assert cc["n_strip_segments"] == 10, cc                    # lead + 8 sides + extension
    assert cc["underpass_transition"]["right_angles"] == 0

    # --- K5: the attribution FAILS, FDFD - referee throughout --------------
    att = d["attribution"]
    assert att["n_corners"] == 8
    assert att["sign_convention"].startswith("FDFD - referee"), att["sign_convention"]
    assert g["K5"]["passed"] is False and att["passed"] is False
    assert "K5 FAILS" in g["K5"]["statement"], g["K5"]["statement"]
    fa = att["finest_admitted"]
    assert fa["cells_across_width"] == 2.0
    assert abs(fa["per_corner_discrepancy"] - 0.1313e-12) <= 1e-16, fa     # POSITIVE
    assert abs(fa["per_corner_uncertainty"] - 1.0251e-12) <= 1e-16, fa
    assert fa["per_corner_uncertainty_dominated"] is True
    assert abs(fa["corners_total"] - 1.0505e-12) <= 1e-16, fa
    assert abs(fa["corners_total_uncertainty"] - 8.2012e-12) <= 1e-16, fa
    lo, hi = fa["corners_interval"]
    assert abs(lo + 7.1506e-12) <= 1e-16 and abs(hi - 9.2517e-12) <= 1e-16, (lo, hi)
    # the three residuals (all NEGATIVE: the FDFD spiral is low) and the misses
    for key, t_lo, t_hi, miss in (
            ("vs_continuum_limit", -15.1813, -8.2489, 1.0983),                  # primary
            ("vs_continuum_limit_after_corrections", -15.4739, -8.4415, 1.2908),
            ("vs_old_remainder", -19.1913, -8.4094, 1.2588)):                   # superseded
        c = fa[key]
        assert abs(c["target_range"][0] - t_lo * 1e-12) <= 1e-16, (key, c)
        assert abs(c["target_range"][1] - t_hi * 1e-12) <= 1e-16, (key, c)
        assert c["intersects"] is False and c["overlap_range"] is None, (key, c)
        assert abs(c["miss"] - miss * 1e-12) <= 1e-16, (key, c)
        # the miss is the corner interval's LOWER end above the residual's UPPER end
        assert c["miss"] == lo - c["target_range"][1], (key, c)
        assert c["central_value_inside"] is False, (key, c)
        assert c["central_value_same_sign"] is False, (key, c)          # the sign error
    assert fa["vs_continuum_limit"]["target_range"] == att["primary"]["continuum_limit_range"]
    assert (fa["vs_continuum_limit_after_corrections"]["target_range"]
            == att["primary"]["after_corrections_range"])
    # the primary comes from the invariant ladder, keys stated
    pr = att["primary"]
    assert pr["source"] == "validation/fdfd/invariant_ladder.json", pr["source"]
    assert pr["keys"] == {"referee": "referee.total",
                          "continuum_limit_rel_range": "residual.limit_rel_range",
                          "after_corrections_rel_range":
                              "residual.after_corrections_rel_range"}, pr["keys"]
    assert pr["referee"] == att["spiral_referee_deembedded"] == 3.3305510896906523e-10
    # the superseded construction, signed: D2's residual minus D2b's deficit
    old = att["secondary_old_remainder"]
    assert abs(old["spiral_residual_range"][0] + 25.5859e-12) <= 1e-16, old
    assert abs(old["spiral_residual_range"][1] + 16.8690e-12) <= 1e-16, old
    assert abs(old["bar_explained_range"][0] + 8.4596e-12) <= 1e-16, old
    assert abs(old["bar_explained_range"][1] + 6.3947e-12) <= 1e-16, old
    assert old["remainder_range"] == fa["vs_old_remainder"]["target_range"]
    # the excluded raw pad-4 W/3 value is further still, and of the same wrong sign
    raw5 = g["K5"]["finest_raw_pad4_EXCLUDED"]
    assert raw5["cells_across_width"] == 3.0
    assert abs(raw5["corners_total"] - 25.7751e-12) <= 1e-16, raw5
    assert raw5["vs_continuum_limit"]["intersects"] is False
    assert abs(raw5["vs_continuum_limit"]["miss"] - 34.0240e-12) <= 1e-16, raw5

    # --- K_area and the bridge-convention insensitivity --------------------
    assert g["K_area"]["passed"]
    sens = d["arm_sensitivity"]["2"]
    assert abs(sens["excess_spread_rel"] - 0.00194) <= 1e-4, sens
    assert sens["excess_spread_rel"] < 0.01, "the corner excess must not depend on the bridge"

    # --- the recorded referee is level-consistent --------------------------
    for dk, row in exc.items():
        assert row["excess_referee"] < 0.0
        assert abs(row["excess_referee"] + 7.78e-12) <= 2e-14, (dk, row["excess_referee"])
        assert row["referee"]["bend"]["L_strip"] > row["L_referee"]["bend"]
        assert row["referee"]["bar"]["L_strip"] > row["L_referee"]["bar"]
    assert abs(d["seconds_measuring"] / 60.0 - 34.34) <= 0.05, d["seconds_measuring"]


def test_gate_k3_is_evaluated_on_the_admitted_levels_only():
    """T9. K3 gates the sign of the corner excess on the ADMITTED levels
    only -- W/1 and W/2, the set K4 and K5 use -- on their wall-corrected
    values, and reports W/3 (no three-point ladder; its raw pad-4 value) as
    excluded information. The recorded gate is re-evaluated from the JSON
    with the study's own ``evaluate_gates``, and two sign flips show the
    restriction is in the code: a W/3 of the wrong sign leaves K3 passing,
    an admitted level of the wrong sign fails it. No solver."""
    st, d = _study(), _json()
    k3 = d["gates"]["K3"]
    assert k3["admitted_levels"] == ["W/1", "W/2"], k3["admitted_levels"]
    assert k3["admitted_levels"] == d["gates"]["K1"]["admitted_levels"]
    assert k3["admitted_levels"] == d["excess"]["admitted_levels"]
    assert [r["div"] for r in d["gates"]["K4"]["admitted"]] == [1.0, 2.0]
    assert d["attribution"]["finest_admitted"]["div"] == 2.0
    per = k3["per_level"]
    assert list(per) == ["W/1", "W/2"], list(per)
    for key, dk, f_, r_ in (("W/1", "1", -7.2611, -7.7934), ("W/2", "2", -7.6533, -7.7846)):
        v = per[key]
        assert v["excess_fdfd"] == d["excess"]["levels"][dk]["excess_fdfd"]   # wall-corrected
        assert abs(v["excess_fdfd"] - f_ * 1e-12) <= 1e-16, v
        assert abs(v["excess_referee"] - r_ * 1e-12) <= 1e-16, v
        assert v["sign_fdfd"] == -1 and v["sign_referee"] == -1 and v["agrees"] is True, v
    assert k3["passed"] is True

    ex = k3["excluded_informational"]
    assert list(ex) == ["W/3"], list(ex)
    w3 = ex["W/3"]
    assert w3["excess_fdfd"] == d["excess"]["levels"]["3"]["excess_fdfd_pad4"]
    assert abs(w3["excess_fdfd"] + 4.5607e-12) <= 1e-16, w3
    assert abs(w3["excess_referee"] + 7.7826e-12) <= 1e-16, w3
    assert abs(w3["ratio"] - 0.5860) <= 1e-4, w3
    assert w3["sign_fdfd"] == -1 and w3["agrees"] is True, w3

    # the recorded gate is what the code gives on the recorded study ...
    assert st.evaluate_gates(d)["K3"] == k3
    # ... a W/3 of the WRONG sign does not move the gate ...
    flip3 = copy.deepcopy(d)
    flip3["excess"]["levels"]["3"]["excess_fdfd_pad4"] *= -1.0
    g3 = st.evaluate_gates(flip3)["K3"]
    assert g3["passed"] is True, g3
    assert g3["excluded_informational"]["W/3"]["agrees"] is False, g3
    assert abs(g3["excluded_informational"]["W/3"]["excess_fdfd"] - 4.5607e-12) <= 1e-16
    # ... and an admitted level of the wrong sign fails it
    flip2 = copy.deepcopy(d)
    flip2["excess"]["levels"]["2"]["excess_fdfd"] *= -1.0
    g2 = st.evaluate_gates(flip2)["K3"]
    assert g2["passed"] is False, g2
    assert g2["per_level"]["W/2"]["agrees"] is False, g2


def test_gate_k5_rederives_from_the_json_and_the_invariant_ladder(tmp_path):
    """T10. The study's own ``attribution``, re-run on the recorded excess
    and corner count, reads ``invariant_ladder.json`` itself and reproduces
    the recorded block and gate K5 exactly. Its primary residual is that
    JSON's ``residual.limit_rel_range`` x ``referee.total``, and an
    invariant ladder whose referee is not D2's is refused. No solver."""
    st, d = _study(), _json()
    il = json.loads(INVARIANT_PATH.read_text())
    att = json.loads(json.dumps(st.attribution(d["excess"], d["corner_count"])))
    assert att == d["attribution"]
    assert json.loads(json.dumps(st.evaluate_gates(d)["K5"])) == d["gates"]["K5"]

    ref = il["referee"]["total"]
    assert ref == st.SPIRAL_REFEREE == 3.3305510896906523e-10
    pr = att["primary"]
    assert pr["continuum_limit_rel_range"] == il["residual"]["limit_rel_range"]
    assert pr["after_corrections_rel_range"] == il["residual"]["after_corrections_rel_range"]
    assert pr["continuum_limit_range"] == [v * ref for v in il["residual"]["limit_rel_range"]]
    assert abs(pr["continuum_limit_rel_range"][0] + 0.045582) <= 1e-6, pr
    assert abs(pr["continuum_limit_rel_range"][1] + 0.024767) <= 1e-6, pr
    assert abs(pr["after_corrections_rel_range"][0] + 0.046461) <= 1e-6, pr
    assert abs(pr["after_corrections_rel_range"][1] + 0.025346) <= 1e-6, pr
    # the invariant ladder's own P3 gate carries the same range
    assert pr["continuum_limit_rel_range"] == il["gates"]["P3"]["rel_range"]

    bad = copy.deepcopy(il)
    bad["referee"]["total"] = 1.01 * ref
    path = tmp_path / "invariant_ladder.json"
    path.write_text(json.dumps(bad))
    with pytest.raises(ValueError):
        st.load_invariant_ladder(path)
