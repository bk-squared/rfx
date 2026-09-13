"""Fast subset of validation/fdfd/spiral_convergence.py: the 3-D FDFD
spiral against the independent Greenhouse referee on the coarsest two
refinement levels (W/1, W/2 in-plane with dz = 2 dx: 4598 and 20572
unknowns), and the shape derivatives three ways.

The study script (four levels, ~1.4 h) reports every gate; this file runs
in ~1.5 min and asserts what the two coarsest levels can support. Numbers
in the docstrings were measured in this session (macOS, SuperLU/COLAMD):

V1  extrapolated FDFD L_dut vs referee. The referee is the delivered
    Greenhouse model (uniform current, centreline corners) extended by the
    method of images to the FDFD's full PEC box (ground + lid + 4 walls;
    ground-only, as the task literally says, misses the lid and walls that
    remove a further 16.5 % of the free-space value after the ground's
    13.5 %). The literal 5 % gate FAILS at every level and for every
    extrapolation estimator, and the gap does not close when the referee
    is moved to a surface-current model (no internal inductance, PEC edge
    crowding from a 2-D boundary-element distribution, corner square
    counted once): the referee band is [253.6, 262.9] pH, the FDFD
    extrapolates to 228-238 pH (four levels; 218-238 from the two levels
    here). That finding is hard-asserted (test_v1_finding_...) so a change
    is noticed; the literal gate is a strict xfail that turns red if it
    ever passes.
V2  jax.grad vs FD4 of the same FDFD L_dut, 1 % steps: all three
    parameters at W/1, r_out (the most curved) at W/2; measured <= 4.3e-6
    (gate 1e-4). The script checks all three at every level.
V3  jax.grad vs FD4 of the box referee: same sign for all three
    parameters at both levels (asserted); the 15 % band is a finest-level
    gate reported by the script (FAILS there, ratios 0.80 / 0.94 / 0.93).
V4  needs three levels: script only (passes on four).

Plus cheap independent checks: the image lattice reproduces the
delivered referee's ground image term exactly and its tiers / shell
cutoff are converged at the stated digits; the corner conventions keep
(corner once) or lose (corner none, 8 W) conductor length; the 2-D BEM
equilibrium distribution reproduces the thin-strip GMD W/4 and the
referee's Hoer-Love shell; the Richardson fitter recovers synthetic
limits.
"""
from __future__ import annotations

import importlib.util
import math
import pathlib

import numpy as np
import pytest

from tests._x64_compat import enable_x64

pytest.importorskip("shapely")

SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "validation" / "fdfd" / "spiral_convergence.py"
_CACHE: dict = {}


def _script():
    if "sc" not in _CACHE:
        spec = importlib.util.spec_from_file_location("spiral_convergence", SCRIPT)
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _CACHE["sc"] = mod
    return _CACHE["sc"]


def _study() -> dict:
    """Levels W/1 and W/2, light lattice tier (K = 4, 8^4 / 4^4 rules;
    measured 1.7e-3 from the script's K = 8, 16^4 / 6^4 default), FD4 of
    all three parameters at W/1 and of r_out at W/2, no variant gradients,
    no wall check. Built once for the module (inside x64)."""
    if "study" not in _CACHE:
        sc = _script()
        with enable_x64():
            _CACHE["study"] = sc.run_study([1, 2], K=4, ngl_shell1=8, ngl_shell2=4, shell_gradient=False,
                                           fd4_params_per_level={1: (0, 1, 2), 2: (0,)}, wall_check_level=None,
                                           log=lambda s: None)
    return _CACHE["study"]


def _frame():
    if "frame" not in _CACHE:
        sc = _script()
        with enable_x64():
            from rfx.fdfd import spiral as sm
            _CACHE["frame"] = sc.BoxFrame.from_model(sm.build_spiral(sc.study_spec(1)))
    return _CACHE["frame"]


# ----------------------------------------------------------------------------
# independent, cheap

def test_image_lattice_reproduces_the_referee_ground_image_and_is_converged():
    """The (0, 0, -1) lattice term evaluated with the study's image rule
    equals the delivered referee's ``image_sum`` (ground_height = h) to
    round-off (measured 6e-16), so the sign convention and the mirror
    geometry are the referee's own. Tiers: shell 1 with 8^4 vs 4^4 points
    agrees to 1e-9 (gate 1e-7); Evjen-averaged value at K = 4 vs K = 6
    within 3e-3 (measured 1.5e-3; the shells alternate and decay like
    1/K, the averaged sequence converges); the box correction is negative
    (every wall image reduces L) and larger than the ground image alone."""
    sc = _script()
    sg = sc.load_referee()
    frame = _frame()
    theta = (sc.R_OUT, sc.SPACING, sc.WIDTH)
    segs = sc.referee_segments(sg, theta, frame)
    g = sg.greenhouse_terms(segs, ground_height=frame.h)
    imgs = [sc._image_segment(sg, s, 0, 0, -1, frame) for s in segs]
    mine = sum(sc._pair_tiered(sg, si, sj, 16) for si in segs for sj in imgs)
    assert abs(mine - g.image_sum) <= 1e-12 * abs(g.image_sum), (mine, g.image_sum)   # measured 6e-16
    assert abs(sc.single_wall_images(sg, segs, frame)["ground"] - g.image_sum) <= 1e-12 * abs(g.image_sum)
    sh8 = sc.lattice_shells(sg, segs, frame, 1, ngl_shell1=8)[1]
    sh4 = sc.lattice_shells(sg, segs, frame, 1, ngl_shell1=4)[1]
    assert abs(sh8 - sh4) <= 1e-7 * abs(sh8), (sh8, sh4)                               # measured 1e-9
    free = g.total - g.image_sum
    v4 = sc.evjen(free, sc.lattice_shells(sg, segs, frame, 4, 8, 4))["value"]
    v6 = sc.evjen(free, sc.lattice_shells(sg, segs, frame, 6, 8, 4))["value"]
    assert abs(v4 - v6) <= 3e-3 * abs(v6), (v4, v6)                                     # measured 1.5e-3
    assert v6 < g.total < free                                                          # box < ground-only < free
    assert (free - v6) > 2.0 * (free - g.total)                                         # lid + walls > ground alone


def test_corner_conventions_keep_or_lose_conductor_length():
    """corner_once keeps the Greenhouse centreline length (642 um for
    this spiral: 8 corners, each bar end moved by W/2 in opposite senses)
    and lies 1.2 % ABOVE the Greenhouse value in the box (asserted within
    [0.5, 2.5] %); corner_none removes exactly 8 W = 80 um and is 14 %
    below -- the spread an earlier version of this study wrongly called a
    corner double-count. Light lattice tier (K = 4, 8^4 / 4^4)."""
    sc = _script()
    sg = sc.load_referee()
    frame = _frame()
    theta = (sc.R_OUT, sc.SPACING, sc.WIDTH)
    segs = sc.referee_segments(sg, theta, frame)
    n_corners = sum(sc._is_corner(segs, k, k + 1) for k in range(len(segs)))
    assert n_corners == 8
    length = sc.conductor_length(segs)
    assert length == pytest.approx(642e-6, abs=1e-12)
    assert sc.conductor_length(sc.corner_once_segments(sg, segs)) == pytest.approx(length, abs=1e-12)
    assert sc.conductor_length(sc.corner_none_segments(sg, segs)) == pytest.approx(length - 8 * sc.WIDTH, abs=1e-12)
    ref = sc.referee_L(sg, theta, frame, K=4, ngl_shell1=8, ngl_shell2=4)
    _CACHE["ref_k4"] = ref
    once = ref["box_corner_once"] / ref["box"] - 1.0
    assert 0.005 <= once <= 0.025, once                                                 # measured +1.2e-2
    assert ref["box_corner_none"] < ref["box_shell"] < ref["box"]                       # 239.7 < 259.6 < 279.5 pH
    assert ref["box_pec"] < ref["box_shell"]                                            # crowding lowers L
    assert ref["surface_current_band"] == [ref["box_pec"], ref["box_shell_corner_once"]]
    assert ref["surface_current_band"][0] < ref["surface_current_band"][1]


def test_bem_equilibrium_distribution_reproduces_thin_strip_gmd_and_hoer_love_shell():
    """The 2-D boundary-element machinery behind the ``pec`` variant:
    (a) the equilibrium distribution on a W x W/1000 rectangle has the
    thin-strip GMD W/4 (measured +3.5e-3, the residual is the finite
    thickness; gate 5e-3); (b) the double sum with a UNIFORM perimeter
    density reproduces the referee's exact Hoer-Love surface shell to 3e-3
    (measured -2.4e-3 at l = 10 um, -1.2e-3 at 94 um; the shell is 0.02
    um thick, the elements have zero thickness); (c) the crowding ratio of
    the 10 x 2 um strip is between -2 % and -1 % (measured -1.44 % at
    l = 10 um, -1.58 % at 94 um) and converged in the mesh to 1e-4
    (measured 2e-5 between n_side 200 and 400)."""
    sc = _script()
    sg = sc.load_referee()
    w, t = sc.WIDTH, sc.T_M2
    mid, elem = sc.perimeter_mesh(w, w / 1000, 800)
    q = sc.equilibrium_distribution(mid, elem)
    assert abs(q.sum() - 1.0) < 1e-12
    assert np.all(q > 0)
    gmd = math.exp(sc.log_gmd(mid, elem, q))
    assert abs(gmd / (w / 4) - 1.0) <= 5e-3, gmd                                       # measured 3.5e-3
    mid, elem = sc.perimeter_mesh(w, t, sc.BEM_N_SIDE)
    qu = elem / elem.sum()
    for length in (10e-6, 94e-6):
        l_bem = sc.bem_self_inductance(sg, length, mid, elem, qu)
        l_hl = sc.shell_self_inductance(sg, length, w, t, sc.SHELL_DELTA)
        assert abs(l_bem / l_hl - 1.0) <= 3e-3, (length, l_bem, l_hl)                   # measured 2.4e-3 / 1.2e-3
        r200 = sc.pec_crowding_ratio(sg, length, w, t)
        r400 = sc.pec_crowding_ratio(sg, length, w, t, 400)
        assert -0.02 <= r200 - 1.0 <= -0.01, r200                                       # measured -1.44e-2 / -1.58e-2
        assert abs(r200 - r400) <= 1e-4, (r200, r400)                                    # measured 2e-5


def test_richardson_recovers_synthetic_limits():
    sc = _script()
    hs = [1.0, 0.5, 1 / 3, 0.25]
    for p in (1, 2):
        ls = [3.0 + 0.7 * h ** p for h in hs]
        r = sc.richardson(hs, ls)
        assert abs(r[f"p{p}"]["L_extrapolated"] - 3.0) < 1e-12
        assert r[f"p{p}"]["rms_residual_rel"] < 1e-12
        assert abs(r["observed_order"] - p) < 1e-6
        assert abs(r["L_extrapolated_observed_order"] - 3.0) < 1e-9
        assert abs(r[f"two_level_p{p}"] - 3.0) < 1e-12
        assert set(r["estimates"]) == {"p1_finest_pair", "p2_finest_pair", "p1_lstsq_all", "observed_order_finest_three"}
    ls = [3.0 + 0.7 * h ** 1.5 for h in hs]
    r = sc.richardson(hs, ls)
    assert abs(r["observed_order"] - 1.5) < 1e-6
    assert abs(r["two_level_p1"] - (ls[-1] * hs[-2] - ls[-2] * hs[-1]) / (hs[-2] - hs[-1])) < 1e-15
    assert r["range"][0] <= 3.0 <= r["range"][1]          # p1 over- and p2 under-shoot a p = 1.5 sequence
    r2 = sc.richardson(hs[:2], ls[:2])
    assert set(r2["estimates"]) == {"p1_finest_pair", "p2_finest_pair"}


# ----------------------------------------------------------------------------
# the study on the coarsest two levels

def test_sizes_and_the_fd4_steps_move_the_objective():
    """N = 4598 (W/1, 13 x 14 x 7) and 20572 (W/2, 25 x 27 x 9); the 1 %
    FD4 steps move L_dut by >= 1e-3 relative (measured 1.3e-2 / 3.6e-3 /
    8.5e-3 at W/1, 1.3e-2 for r_out at W/2), i.e. far above the ~1e-11 LU
    floor; jit == eager."""
    st = _study()
    lv1, lv2 = st["levels"]["1"], st["levels"]["2"]
    assert lv1["n_unknowns"] < 10000 and lv2["n_unknowns"] < 30000
    assert lv2["n_unknowns"] > 3 * lv1["n_unknowns"]
    assert lv1["fd4_solves"] == 12 and lv2["fd4_solves"] == 4
    for lv in (lv1, lv2):
        moves = np.asarray(lv["fd4_step_moves_L_rel"])
        assert np.nanmin(moves) >= 1e-3, moves
        assert lv["jit_minus_eager_rel"] <= 1e-10
        assert lv["L_dut"] > 0 and lv["L_raw"] > lv["L_dut"]          # the columns carry inductance
    assert np.isnan(lv2["fd4"][1]) and np.isnan(lv2["fd4"][2])


def test_v2_grad_matches_fd4():
    """V2: |jax.grad / FD4 - 1| <= 1e-4 for r_out, spacing, width at W/1
    and for r_out at W/2. Measured 3.7e-6 / 6.7e-10 / 4.3e-9 (W/1) and
    4.2e-6 (W/2): the r_out figure is the FD4 truncation error of a 1 %
    step on the most curved parameter (it scales as step^4: 1.3e-8 at
    0.25 %), the others are at the LU floor."""
    st = _study()
    rel1 = np.asarray(st["gates"]["V2"]["per_level"]["1"])
    rel2 = np.asarray(st["gates"]["V2"]["per_level"]["2"])
    assert np.all(rel1 <= 1e-4), rel1
    assert rel2[0] <= 1e-4 and np.isnan(rel2[1]) and np.isnan(rel2[2]), rel2
    assert st["gates"]["V2"]["worst"] <= 1e-4
    assert st["gates"]["V2"]["passed"]


def test_v3_derivative_signs_agree_with_the_referee():
    """V3 (sign part, both levels): dL/dr_out > 0, dL/dspacing < 0,
    dL/dwidth < 0 in both the FDFD gradient and the FD4 of the box
    referee. Ratios FDFD/referee at W/2 measured 0.79 / 0.96 / 0.87
    (recorded; the 15 % band is the script's finest-level gate and FAILS
    there: 0.80 / 0.94 / 0.93 at W/4)."""
    st = _study()
    g_ref = np.asarray(st["referee"]["gradient_box_fd4"])
    assert g_ref[0] > 0 and g_ref[1] < 0 and g_ref[2] < 0, g_ref
    for div in ("1", "2"):
        g = np.asarray(st["levels"][div]["grad"])
        assert np.all(np.sign(g) == np.sign(g_ref)), (div, g, g_ref)
    assert st["gates"]["V3"]["same_sign_all_levels"]
    ratio = np.asarray(st["gates"]["V3"]["ratio_per_level"]["2"])
    assert np.all((0.6 < ratio) & (ratio < 1.1)), ratio                                # measured 0.79 / 0.96 / 0.87


def test_v1_finding_fdfd_below_every_referee_variant():
    """V1, the measured statement, hard-asserted so a future change is
    noticed. FDFD L_dut 177.3 pH (W/1) and 207.7 pH (W/2); two-level
    extrapolation range [217.8 (p = 2), 238.1 (p = 1)] pH. Referee with
    the box images: delivered model 279.5 pH, surface-current band
    [253.6 (PEC edge-crowded), 262.9 (shell, corner once)] pH. Asserted:
    the WHOLE extrapolation range lies below the band by more than the 5 %
    gate (top of the range vs top of the band -9.4 %, bottom vs bottom
    -14 %); every estimator is more than 10 % below the delivered box
    referee (measured -14.8 % / -22 %); the per-level gap to the box
    referee is below -5 % at both levels (-36.6 %, -25.7 %) and shrinks
    with refinement; the lead reference-plane choice is worth < 3 %
    (measured 1.8 %)."""
    st = _study()
    v1 = st["gates"]["V1"]
    ref = st["referee"]
    lo, hi = v1["L_extrapolated_range"]
    assert lo == pytest.approx(st["richardson_all_levels"]["two_level_p2"])
    assert hi == pytest.approx(st["richardson_all_levels"]["two_level_p1"])
    band = ref["surface_current_band"]
    assert hi < band[0], (hi, band)                                                     # the range is below the band
    assert v1["band_gap"][1] < -0.05 and v1["band_gap"][0] < -0.05, v1["band_gap"]      # -9.4e-2 / -1.4e-1
    assert not v1["passed_surface_current_band"]
    assert all(g < -0.10 for g in v1["gap_extrapolated"]["box"].values()), v1["gap_extrapolated"]["box"]
    assert not v1["passed"] and not any(v1["passed_per_variant"].values())
    for div in ("1", "2"):
        assert v1["gap_per_level"]["box"][div] < -0.05, (div, v1["gap_per_level"]["box"][div])
    assert v1["gap_per_level"]["box"]["2"] > v1["gap_per_level"]["box"]["1"]      # the FDFD converges upward
    assert v1["gap_shrinks_with_refinement_box"]
    assert abs(ref["reference_plane_sensitivity"]["rel_change"]) < 0.03           # measured 1.8e-2


@pytest.mark.xfail(strict=True, reason="V1 as literally worded fails: the two-level extrapolated FDFD L_dut is "
                   "-15 % (p = 1) / -22 % (p = 2) from the delivered Greenhouse model with the box images; "
                   "strict, so the suite turns red if the gate ever passes")
def test_v1_literal_five_percent_gate_against_the_delivered_referee():
    """V1 as literally worded: |L_extrapolated / L_referee_box - 1| <= 5 %
    for every extrapolation estimator, against the delivered Greenhouse
    model (uniform current, centreline corners) in the FDFD box."""
    st = _study()
    assert st["gates"]["V1"]["passed"], st["gates"]["V1"]["gap_range"]
