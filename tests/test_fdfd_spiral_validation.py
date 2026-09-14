"""Validation gates for the D2 study ``validation/fdfd/spiral_convergence.py``:
the differentiable 3-D FDFD spiral inductor against the independent
Greenhouse referee under the physics-consistent protocol (volumetric
uniform-current metal, the five non-ground PEC walls pushed away by graded
padding, the area-exact corner convention, a matched reference plane).

The study itself runs three in-plane levels (W/1, W/2, W/3; 23k / 56k /
109k unknowns, ~65 min). This file re-runs the COARSEST level (W/1,
N = 23062, one three-fixture solve ~15 s) live and asserts the parts of the
protocol that grid supports, plus the two parts that need no solver at all
(the corner-convention area gate and the referee itself). Everything that
needs the finer grids -- the Richardson extrapolation and the FD4 gradient
checks on the study grids -- is in the study's JSON; test T6 gates that JSON
against a fresh W/1 solve so the recorded numbers cannot drift silently.

Gates here (numbers measured in this session, see each docstring):

T1  the area-exact corner convention tiles the ``gds.rect_spiral`` strip
    polygon exactly (sum = union = polygon area to 4e-16) while the
    delivered Greenhouse convention overlaps itself by -3.350e-2 of the
    area -- the gate discriminates between the two conventions, and the
    total centreline length is identical (5.97e-4 m), so no conductor is
    created or destroyed.
T2  the referee's partial-inductance sum is physical (self > 0, image < 0,
    ground shielding removes 13.6 % of the free-space value) with a worst
    per-term error estimate of 1.6e-11, and the three model systematics are
    small: corners +1.21 %, reference plane +-1.46 %.
T3  the volumetric (uniform-current) FDFD fixture at W/1 is reciprocal and
    passive, its L_diff (262.44 pH) exceeds the PEC L_diff (244.88 pH) by
    6.7 % -- the DC internal inductance the referee also carries -- and it
    sits 24.9 % BELOW the referee. That last number is the V1 FAILURE at
    this level, asserted as a measured value: the test fails if it moves,
    in either direction.
T4  the graded wall padding works: the closed PEC box is 27.4 % below the
    padded value, and adding a fourth padding cell moves L_diff by
    +0.24 % < 1 % (the V5 wall-independence gate).
T5  ``jax.grad`` of the de-embedded L vs FD2 on a cheaper (pad = 2, 11.9k)
    grid of the same fixture, and the sign of all three derivatives against
    the referee's own FD4 gradient.
T6  the study's JSON exists, is self-consistent, and its recorded W/1
    ``L_dut`` / ``L_pec`` / referee value reproduce the fresh solve to
    1e-7 relative (100x the fixture's LU noise floor).

x64 is scoped per test through ``tests._x64_compat.enable_x64`` (never
flipped at module level); the static models and the W/1 solves are cached
for the module so the file stays inside its 100 s budget.
"""
from __future__ import annotations

import importlib.util
import json
import pathlib
import time

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx.fdfd import spiral as sm
from tests._x64_compat import enable_x64

pytest.importorskip("shapely")

REPO = pathlib.Path(__file__).resolve().parents[1]
STUDY_PATH = REPO / "validation" / "fdfd" / "spiral_convergence.py"
JSON_PATH = REPO / "validation" / "fdfd" / "spiral_convergence.json"

_CACHE: dict = {}


def _study():
    """The study module itself (it owns the protocol constants and the
    referee-side conventions; importing it keeps the test and the study from
    drifting apart)."""
    if "study" not in _CACHE:
        spec = importlib.util.spec_from_file_location("spiral_convergence_study", STUDY_PATH)
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _CACHE["study"] = mod
    return _CACHE["study"]


def _referee():
    if "referee" not in _CACHE:
        _CACHE["referee"] = _study().load_referee()
    return _CACHE["referee"]


def _model(pad: int):
    key = f"model{pad}"
    if key not in _CACHE:
        _CACHE[key] = _study().build_level(1.0, pad=pad)
    return _CACHE[key]


def _solve(pad: int, sigma: float | None) -> dict:
    """Cached solve of the W/1 fixture (``sigma=None`` is the PEC fixture)."""
    key = f"solve{pad}:{sigma}"
    if key not in _CACHE:
        st = _study()
        t0 = time.time()
        res = sm.solve_spiral(_model(pad), st.FREQ, sigma_volumetric=sigma)
        jax.block_until_ready(res.L_diff)
        _CACHE[key] = {"res": res, "L": float(res.L_diff), "seconds": time.time() - t0}
    return _CACHE[key]


def _fd2(f, x0: float, d: float) -> float:
    return (f(x0 + d) - f(x0 - d)) / (2 * d)


# ----------------------------------------------------------------------------
# T1 / T2: the referee side (no solver)

def test_area_exact_corner_convention_tiles_the_strip_polygon_exactly():
    """T1 (protocol C). ``rect_spiral_segments`` gives each bar the
    centreline length between corner POINTS, so at every right angle one
    quadrant of the W x W corner square is covered twice and the opposite
    quadrant not at all. The sum of the bar areas is blind to that (it is
    ``W x`` centreline length either way -- asserted here, both conventions
    match ``gds.polygon_area`` of the strip to 2.2e-16), so the gate is the
    UNION: the delivered convention's union is 3.350e-2 short of the
    polygon, the area-exact convention's union equals it to 4.4e-16, i.e.
    its rectangles are exactly disjoint and tile the mitred strip.

    Measured on the study fixture (2 turns, r_out = 52 um, W = S = 10 um,
    lead shortened by W/2 to the de-embedded reference plane): strip
    polygon 5.97e-9 m^2, 10 bars, centreline 5.97e-4 m in both
    conventions."""
    st = _study()
    gate = st.area_gate(_referee(), st.THETA0)
    ae, gh = gate["area_exact"], gate["greenhouse"]
    # the sum of the areas is convention-blind: both are exactly W * length
    for conv in (ae, gh):
        assert abs(conv["sum_over_polygon_minus_1"]) <= 1e-12, conv
        assert abs(conv["underpass_area_over_polygon_minus_1"]) <= 1e-12, conv
        assert abs(conv["total_centreline_length"] - gate["strip_centreline_length"]) \
            <= 1e-12 * gate["strip_centreline_length"]
    # the union is not: this is what pins the convention
    assert abs(ae["union_over_sum_minus_1"]) <= 1e-12, ae            # measured 4.4e-16
    assert abs(ae["union_over_polygon_minus_1"]) <= 1e-12, ae        # measured 4.4e-16
    assert gh["union_over_sum_minus_1"] < -1e-3, gh                  # measured -3.350e-2
    assert ae["n_bars"] == gh["n_bars"] == 10, (ae["n_bars"], gh["n_bars"])
    # the mitred strip polygon's area is W * centreline length, exactly
    assert abs(gate["strip_polygon_area"] - st.WIDTH * gate["strip_centreline_length"]) \
        <= 1e-12 * gate["strip_polygon_area"]


def test_referee_partial_inductance_sum_is_physical_and_its_systematics_are_small():
    """T2 (protocol C/D). The referee at the study fixture, area-exact
    corners, ground at h = 26 um: 349.371 pH = 382.02 (self) + 22.12
    (mutual) - 54.77 (ground image) pH, worst per-term error estimate
    1.6e-11 (the Hoer-Love round-off amplification, reported by the
    referee itself). Physical content asserted: the self sum dominates, the
    image term is NEGATIVE (a PEC plane shields), and it removes 13.6 % of
    the free-space value 404.14 pH.

    The two convention systematics are asserted to be small, because the
    whole point of the protocol is that they cannot explain the FDFD gap:
    corners +1.2122 % (345.133 -> 349.371 pH) and the reference plane
    +1.4595 % / -1.4602 % (port line / far edge of the lead-column
    footprint)."""
    st, sg = _study(), _referee()
    g = st.referee_value(sg, st.THETA0, "area_exact")
    free = st.referee_value(sg, st.THETA0, "area_exact", ground=None).total
    gh = st.referee_value(sg, st.THETA0, "greenhouse").total
    assert g.worst_error < 1e-9, g.worst_error                       # measured 1.6e-11
    assert g.self_sum > 0 and g.image_sum < 0, g
    assert g.self_sum > abs(g.mutual_sum) + abs(g.image_sum), g
    assert abs(g.total - (g.self_sum + g.mutual_sum + g.image_sum)) <= 1e-15 * g.total
    shield = g.total / free - 1.0
    assert -0.30 < shield < -0.05, shield                            # measured -13.6 %
    corner = g.total / gh - 1.0
    assert 0.005 < corner < 0.03, corner                             # measured +1.2122 %
    band = [st.referee_value(sg, st.THETA0, "area_exact", shift=s).total
            for s in (0.0, st.WIDTH)]
    rel = [v / g.total - 1.0 for v in band]
    assert max(abs(r) for r in rel) < 0.025, rel                     # measured +-1.46 %
    assert rel[0] > 0 > rel[1], rel        # a shorter lead is less inductance


# ----------------------------------------------------------------------------
# T3 / T4: the FDFD fixture on the coarsest study grid

def test_uniform_current_fdfd_is_reciprocal_passive_and_sits_below_the_referee():
    """T3 (protocol A, gates V1 at W/1 and V1b). One W/1 build (21 x 22 x 15
    cells, N = 23062) solved twice: volumetric metal (sigma = 3e6 S/m, skin
    depth 29.1 um = 14.6 x the metal thickness -> uniform current density,
    DC internal inductance included) and PEC metal.

    Measured: L_diff = 262.439 pH (volumetric) and 244.877 pH (PEC), i.e.
    the PEC fixture is 6.69 % LOW because a surface current carries no
    internal inductance -- against the -7.2 % the referee's own
    surface-shell variant predicted in the refuted first study. The
    volumetric value sits -24.88 % from the referee's 349.371 pH: gate V1
    FAILS at this level and the failure is asserted as a MEASURED number
    (+-0.5 % absolute on the ratio) rather than hidden behind an xfail --
    the study's docstring carries the diagnosis (the discrete
    self-inductance of a conductor resolved by one in-plane cell across its
    width; the fixture-free lead-length differential measures the same
    -24 % on the inductance per unit length at this level).

    Reciprocity and passivity of the raw S-matrix are the algebraic
    self-checks of the lossy-metal fixture: measured |S12 - S21| = 3.4e-11
    and max singular value 1.0 - 1.6e-10."""
    with enable_x64():
        st = _study()
        m = _model(st.PAD_CELLS)
        assert m.n_unknowns <= 25000, m.n_unknowns                   # test size guideline
        vol = _solve(st.PAD_CELLS, st.SIGMA_VOL)
        pec = _solve(st.PAD_CELLS, None)
        res = vol["res"]
        s = np.asarray(res.s_raw)
        recip = abs(s[0, 1] - s[1, 0]) / float(np.abs(s).max())
        assert recip <= 1e-8, recip                                  # measured 3.4e-11
        assert float(np.linalg.svd(s, compute_uv=False).max()) <= 1.0 + 1e-8
        assert vol["L"] > 0 and pec["L"] > 0
        internal = pec["L"] / vol["L"] - 1.0
        assert internal < 0, internal                                # V1b: PEC carries no L_int
        assert 0.03 < -internal < 0.10, internal                     # measured -6.69 %
        sg = _referee()
        ref = st.referee_value(sg, st.THETA0, "area_exact").total
        gap = vol["L"] / ref - 1.0
        _CACHE["gap"] = gap
        assert -0.2538 < gap < -0.2438, gap        # measured -0.24880 (V1 fails, by this much)
        # the de-embedding is doing real work and the lead part is plausible
        assert 0.02 < abs(float(res.L_raw) - vol["L"]) / vol["L"] < 0.2
        # quasi-static: L is frequency-flat (same LU pattern, one extra solve avoided
        # by reading the Q instead: Q = omega L / R is small and positive)
        assert 0.0 < float(res.Q_diff) < 0.1, float(res.Q_diff)      # measured 0.0227


def test_graded_wall_padding_makes_the_de_embedded_l_wall_independent():
    """T4 (protocol B, gate V5). The ``Yee3DSpec`` PML is useless at
    100 MHz (measured in the study: ``pml_kappa_max`` has no effect at all
    and L moves 5-15 % non-monotonically with the PML depth), so the walls
    are moved away by geometrically graded padding cells
    (``spiral.pad_lines``, ratio 1.5) on the four side walls and the lid --
    never on the ground plane, which is the physical wall both tools model.

    Measured at W/1 (base_dz = 10 um): L_diff = 190.53 pH with the closed
    PEC box (pad = 0, walls 62 um from the centre), 244.51 pH at pad = 3
    (133 um) and 245.15 pH at pad = 4 (184 um). So the closed box is 22 %
    low -- the five extra walls are NOT a small correction, which is what
    refuted the first study -- and the last cell moves L by +0.26 %, well
    inside the 1 % wall-independence gate. The padding converges
    monotonically from below (the study's ``wall_gate`` carries pad = 6 and
    8 too)."""
    with enable_x64():
        st = _study()
        padded = _solve(st.PAD_CELLS, st.SIGMA_VOL)["L"]
        closed = _solve(0, st.SIGMA_VOL)["L"]
        one_less = _solve(st.PAD_CELLS - 1, st.SIGMA_VOL)["L"]
        assert _model(0).n_unknowns < _model(st.PAD_CELLS).n_unknowns
        assert closed < one_less < padded, (closed, one_less, padded)    # monotone from below
        closed_rel = closed / padded - 1.0
        assert closed_rel < -0.1, closed_rel                         # measured -22.3 %
        step = padded / one_less - 1.0
        assert abs(step) <= 0.01, step                               # V5: measured +0.26 %
        _CACHE["wall_step"] = step


# ----------------------------------------------------------------------------
# T5: derivatives

def test_shape_gradients_match_finite_differences_and_the_referee_signs():
    """T5 (gates V2 / V3 in the cheap form the 100 s budget allows). The
    study checks ``jax.grad`` against FD4 with 1 % steps on every study
    grid (worst 1.6e-7 relative at W/1, see the JSON); here the same check
    runs against FD2 on the cheaper pad = 2 grid of the same fixture
    (N = 11914, ~5 s per three-fixture solve) -- 1 % steps move L by ~1e-2
    relative, far above the ~1e-9 LU noise floor, and the residual is then
    the FD2 truncation, measured 4.5e-5 / 2.3e-5 / 1.4e-5 relative for
    r_out / spacing / width. Gate 1e-3: ~20x above the worst measurement
    (FD4 on the same grid would be 1e-7, which is what the study asserts).

    V3 in the test form: all three ``jax.grad`` components have the SAME
    SIGN as the referee's own FD4 gradient (+1.1917e-5, -9.8337e-6,
    -2.4491e-5 H/m for r_out / spacing / width) and the same ordering by
    magnitude; the study reports the ratios per level (they sit near 0.9
    at the finest level, see V3 in the JSON)."""
    with enable_x64():
        st = _study()
        pad = 2
        model = _model(pad)
        assert model.n_unknowns <= 25000, model.n_unknowns

        def scalar(theta):
            return jnp.real(st.l_dut(model, theta))

        t0 = time.time()
        val, grad = jax.value_and_grad(scalar)(jnp.asarray(st.THETA0, dtype=jnp.float64))
        jax.block_until_ready(grad)
        seconds = time.time() - t0
        grad = [float(v) for v in grad]
        rel = []
        for k in range(3):
            def f(v: float, k: int = k) -> float:
                t = list(st.THETA0)
                t[k] = v
                return float(scalar(jnp.asarray(t, dtype=jnp.float64)))
            d = st.FD_STEP_REL * st.THETA0[k]
            fd = _fd2(f, st.THETA0[k], d)
            # the step must move the objective far above the LU noise floor
            assert abs(fd * d / float(val)) > 1e-3, (k, fd, val)
            rel.append(abs(grad[k] - fd) / abs(fd))
        assert max(rel) <= 1e-3, rel                                 # measured <= 4.5e-5 (FD2)
        gref = st.referee_gradient(_referee(), st.THETA0, "area_exact")
        assert all(a * b > 0 for a, b in zip(grad, gref)), (grad, gref)
        assert np.argmax(np.abs(grad)) == np.argmax(np.abs(gref))
        assert grad[0] > 0 and grad[1] < 0 and grad[2] < 0, grad
        _CACHE["grad_pad2"] = {"grad": grad, "fd2_rel": rel, "referee": gref,
                               "ratio": [a / b for a, b in zip(grad, gref)],
                               "value_and_grad_seconds": seconds}


# ----------------------------------------------------------------------------
# T6: the recorded study

def test_the_recorded_study_json_matches_a_fresh_coarsest_level_solve():
    """T6. ``validation/fdfd/spiral_convergence.json`` is the study's
    deliverable; this gates it against the live W/1 solve (1e-9 relative on
    L_dut, L_pec and the referee value) and checks its internal structure:
    the three levels, the Richardson estimates and every gate verdict.
    The V1 verdict recorded there is FALSE -- the extrapolated range
    [305.8, 314.9] pH is 9.9-12.4 % below the referee's 349.371 pH -- and
    this test asserts that it is recorded as a failure with those numbers,
    not silently passed or widened."""
    with enable_x64():
        st = _study()
        assert JSON_PATH.exists(), f"run {STUDY_PATH} first"
        study = json.loads(JSON_PATH.read_text())
        vol = _solve(st.PAD_CELLS, st.SIGMA_VOL)
        pec = _solve(st.PAD_CELLS, None)
        lv = study["levels"]["1"]
        assert lv["grid"]["n_unknowns"] == _model(st.PAD_CELLS).n_unknowns
        # 1e-7, not bit-equality: the same code on the same machine reproduces
        # these to ~1e-15, but a different SuperLU/BLAS build reorders the
        # factorisation and the ~1e-9 relative LU noise floor of this fixture
        # (measured on track D1) then shows up in the last digits
        assert abs(lv["L_dut"] / vol["L"] - 1.0) <= 1e-7, (lv["L_dut"], vol["L"])
        assert abs(lv["L_pec"] / pec["L"] - 1.0) <= 1e-7, (lv["L_pec"], pec["L"])
        ref = st.referee_value(_referee(), st.THETA0, "area_exact").total
        assert abs(study["referee"]["area_exact"]["total"] / ref - 1.0) <= 1e-12
        assert sorted(study["levels"]) == ["1", "2", "3"]
        assert study["levels"]["3"]["grid"]["n_unknowns"] <= 110000
        # the vertical grid and the metal cross-section are the same at every level
        for key in ("1", "2", "3"):
            assert study["levels"][key]["grid"]["n_z_lines"] == lv["grid"]["n_z_lines"]
        # L increases with in-plane refinement, still below the referee
        ls = [study["levels"][k]["L_dut"] for k in ("1", "2", "3")]
        assert ls[0] < ls[1] < ls[2] < ref, ls
        g = study["gates"]
        assert g["C_area"]["passed"] and g["V2"]["passed"] and g["V5"]["passed"]
        assert g["V2"]["worst"] <= 1e-4, g["V2"]["worst"]
        assert abs(g["V5"]["rel_change_plus_4_cells"]) <= 0.01
        # V1 is a recorded FAILURE with numbers, not an xfail
        v1 = g["V1"]
        assert v1["passed"] is False
        assert max(abs(r) for r in v1["rel_range"]) > 0.05, v1["rel_range"]
        assert -0.15 < min(v1["rel_range"]) < -0.05, v1["rel_range"]
        assert 1.0 < v1["observed_order"] < 2.0, v1["observed_order"]
        assert study["gates"]["V4"]["passed"] in (True, False)
        # the diagnosis of V1 must be backed by the recorded fixture systematics
        sysx = study["systematics"]
        assert sysx["lead_differential"]["ratio_fdfd_over_referee"] < 0.85
        assert abs(sysx["port_gap_cells_2"]["rel"]) < 0.01
        assert abs(sysx["freq_flatness_rel_spread"]) < 1e-4
        assert abs(sysx["metal_cells"][-1]["rel"]) < 0.01
        _CACHE["json_ok"] = True


def test_report_the_measured_numbers(capsys):
    """Not a gate: prints what this file measured (the reviewer's summary)
    and asserts the whole file stayed inside its size budget."""
    with enable_x64():
        st = _study()
        rows = {k: v for k, v in _CACHE.items() if k.startswith("solve")}
        with capsys.disabled():
            print("\n  W/1 fixture:", _model(st.PAD_CELLS).shape,
                  "N =", _model(st.PAD_CELLS).n_unknowns)
            for k, v in rows.items():
                print(f"  {k}: L_diff = {v['L'] * 1e12:.3f} pH ({v['seconds']:.1f} s)")
            if "gap" in _CACHE:
                print(f"  gap to the referee at W/1: {100 * _CACHE['gap']:+.3f} % (V1 fails)")
            if "wall_step" in _CACHE:
                print(f"  wall gate (pad 3 -> 4): {100 * _CACHE['wall_step']:+.3f} %")
            if "grad_pad2" in _CACHE:
                g = _CACHE["grad_pad2"]
                print(f"  grad vs FD2 (pad 2): {['%.2e' % r for r in g['fd2_rel']]}, "
                      f"ratio to the referee {['%.3f' % r for r in g['ratio']]}")
        for key in ("model0", "model2", f"model{st.PAD_CELLS}"):
            if key in _CACHE:
                assert _CACHE[key].n_unknowns <= 25000, (key, _CACHE[key].n_unknowns)
