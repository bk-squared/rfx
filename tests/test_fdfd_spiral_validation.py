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
T2b the referee quantity the FIXTURE measures is the strip MINUS the short
    standard's bridge (349.371 - 16.316 = 333.055 pH), and the pairing is
    pinned by an invariance, not by assertion: moving the reference plane
    through the lead-column footprint moves the strip by 2.916 % and the
    DIFFERENCE by 0.0914 %, a 32x collapse.
T3  the volumetric (uniform-current) FDFD fixture at W/1 is reciprocal and
    passive, its L_diff (262.44 pH) exceeds the PEC L_diff (244.88 pH) by
    6.7 % -- the DC internal inductance the referee also carries -- and it
    sits 21.2 % below the de-embedded referee (24.9 % below the strip
    alone). That is the V1 FAILURE at this level, asserted as a measured
    value: the test fails if it moves, in either direction.
T4  the graded wall padding works: the closed PEC box is 31.5 % below the
    padded value, and adding a fourth padding cell moves L_diff by
    +0.50 % < 1 % (the V5 wall-independence gate).
T4b a DUT made identical to the short standard's bridge (``dut_kind="bar"``,
    40 um) de-embeds to +1.621 pH, 9.6 % of the 16.896 pH the referee gives
    for that bar: the de-embedding's zero is a strip between the column
    tops, which is why T2b subtracts one (gate V6).
T5  ``jax.grad`` of the de-embedded L vs FD2 on a cheaper (pad = 2, 11.9k)
    grid of the same fixture, and the sign of all three derivatives against
    the referee's own FD4 gradient of the de-embedded quantity.
T6  the study's JSON exists, is self-consistent, its derived blocks
    (Richardson, vertical budget, every gate verdict) re-derive exactly
    from its recorded raw measurements, and its recorded W/1 ``L_dut`` /
    ``L_pec`` / referee value reproduce the fresh solve to 1e-7 relative
    (100x the fixture's LU noise floor).

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


def _same(a, b, path: str = "") -> None:
    """Deep equality of two JSON-shaped trees, floats to 1e-12 relative.
    Used to re-derive the study's recorded blocks from its recorded raw
    measurements."""
    assert type(a) is type(b) or isinstance(a, (int, float)) and isinstance(b, (int, float)), path
    if isinstance(a, dict):
        assert set(a) == set(b), (path, sorted(set(a) ^ set(b)))
        for k in a:
            _same(a[k], b[k], f"{path}/{k}")
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b), (path, len(a), len(b))
        for i, (x, y) in enumerate(zip(a, b)):
            _same(x, y, f"{path}[{i}]")
    elif isinstance(a, bool) or a is None or isinstance(a, str):
        assert a == b, (path, a, b)
    else:
        assert abs(a - b) <= 1e-12 * max(1e-30, abs(a), abs(b)), (path, a, b)


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


def test_the_referee_quantity_the_fixture_measures_is_the_strip_minus_the_bridge():
    """T2b (protocol E). Open/short de-embedding subtracts the SHORT
    standard's differential path, and that path is not empty: it is the
    bridge from the outer lead column (on M2) to the inner one (on M1)
    through the grounded post. Driven differentially the post carries no
    current, and the lead columns are z-directed while strip and bridge are
    x/y-directed (zero Neumann mutual), so the columns cancel term by term
    and the fixture measures ``L(strip) - L(bridge)``.

    Measured on the study fixture: strip 349.371 pH, bridge 16.316 pH,
    de-embedded referee 333.055 pH -- i.e. the strip-only comparison the
    first version of this study used overstates the FDFD gap by 4.9 %.

    The gate is not the arithmetic (that is trivial) but the INVARIANCE
    that says the pairing is right: the reference plane inside the
    ``W x W`` lead-column footprint cuts the strip in y and the bridge in
    x, and moving it from the port line to the far edge moves the strip by
    2.916 % of its value (the +-1.46 % band of T2) while moving the
    DIFFERENCE by only 0.0914 % -- a 32x collapse. The remaining
    convention freedom, where the post splits the bridge between M2 and
    M1, is 0.205 % over the whole range."""
    st, sg = _study(), _referee()
    b = st.referee_deembedded_block(sg, st.THETA0)
    assert abs(b["total"] - (b["strip"] - b["bridge"])) <= 1e-15 * b["strip"]
    assert 0.03 < b["bridge"] / b["strip"] < 0.07, b                  # measured 4.67 %
    assert b["bridge_worst_error"] < 1e-9, b["bridge_worst_error"]    # measured 1.8e-13
    # the invariance: the difference is 10x better determined than the strip
    assert b["reference_plane_spread"] < 0.1 * b["reference_plane_spread_strip_only"], b
    assert b["reference_plane_spread"] < 0.003, b                     # measured 9.14e-4
    assert b["reference_plane_spread_strip_only"] > 0.02, b           # measured 2.916e-2
    assert b["split_spread"] < 0.005, b                               # measured 2.05e-3
    # and it is monotone in the plane position on the strip but not on the
    # difference -- the two ends of the band bracket the centre value
    pp = b["per_plane"]
    assert pp["port_line"]["strip"] > pp["column_centre"]["strip"] > pp["far_edge"]["strip"]
    assert pp["port_line"]["bridge"] > pp["column_centre"]["bridge"] > pp["far_edge"]["bridge"]
    # the derivative of the de-embedded quantity is what V3 compares against:
    # the bridge does not depend on r_out, but it does on spacing and width
    gd = st.referee_gradient(sg, st.THETA0, "area_exact", True)
    gs = st.referee_gradient(sg, st.THETA0, "area_exact", False)
    assert abs(gd[0] / gs[0] - 1.0) <= 1e-9, (gd[0], gs[0])           # bridge indep. of r_out
    assert abs(gd[1] / gs[1] - 1.0) > 0.05, (gd[1], gs[1])            # measured +10.7 %
    assert abs(gd[2] / gs[2] - 1.0) > 0.01, (gd[2], gs[2])            # measured +1.86 %


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
        dee = st.referee_deembedded(sg, st.THETA0)
        gap = vol["L"] / dee - 1.0
        _CACHE["gap"] = gap
        _CACHE["gap_strip_only"] = vol["L"] / ref - 1.0
        # V1 at this level, against the quantity the FIXTURE measures
        assert -0.2170 < gap < -0.2070, gap        # measured -0.21203 (V1 fails, by this much)
        # and against the strip alone, which is what the refuted first
        # version compared: 4.9 % more gap, all of it de-embedding convention
        assert -0.2538 < _CACHE["gap_strip_only"] < -0.2438, _CACHE["gap_strip_only"]   # -0.24880
        assert _CACHE["gap_strip_only"] < gap, (_CACHE["gap_strip_only"], gap)
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

    Measured at W/1 (base_dz = 10 um, volumetric metal): L_diff = 179.899 pH
    with the closed PEC box (pad = 0, walls 62 um from the centre),
    261.124 pH at pad = 3 (133 um) and 262.439 pH at pad = 4 (184 um). So
    the closed box is 31.5 % low -- the five extra walls are NOT a small
    correction, which is what refuted the first study -- and the last cell
    moves L by +0.504 %, inside the 1 % wall-independence gate. The padding
    converges monotonically from below (the study's ``wall_gate`` carries
    pad = 6 and 8 too: 262.731 and 262.740 pH, i.e. +0.114 % from pad 4 to
    pad 8, which is the V5 number)."""
    with enable_x64():
        st = _study()
        padded = _solve(st.PAD_CELLS, st.SIGMA_VOL)["L"]
        closed = _solve(0, st.SIGMA_VOL)["L"]
        one_less = _solve(st.PAD_CELLS - 1, st.SIGMA_VOL)["L"]
        assert _model(0).n_unknowns < _model(st.PAD_CELLS).n_unknowns
        assert closed < one_less < padded, (closed, one_less, padded)    # monotone from below
        closed_rel = closed / padded - 1.0
        assert closed_rel < -0.1, closed_rel                         # measured -31.5 %
        step = padded / one_less - 1.0
        assert abs(step) <= 0.01, step                               # V5: measured +0.504 %
        _CACHE["wall_step"] = step


# ----------------------------------------------------------------------------
# T4b: what the de-embedding's zero is

def test_a_dut_identical_to_the_short_bridge_de_embeds_to_nearly_zero():
    """T4b (protocol E, gate V6). The DIRECT measurement behind the
    strip-minus-bridge correction, with no referee in the loop for the
    claim itself: build the SAME fixture with ``dut_kind="bar"`` and
    ``bar_length`` equal to the spiral's lead-column separation (40 um), so
    the DUT occupies exactly the M2 cells the short standard's bridge and
    post occupy. If the de-embedded ``L_diff`` were "the inductance of the
    conductor between the two reference planes", this fixture would return
    the referee value of that bar; if the de-embedding instead subtracts
    the bridge, it must return ~0.

    Measured at W/1 (15 x 11 x 15, N = 8576, ~4 s): raw L = 24.367 pH,
    de-embedded L = +1.621 pH, while the referee for that bar (self +
    PEC-ground image, h = 26 um) is 16.896 pH. So the de-embedding removes
    90.4 % of the bar and 93.3 % of the raw value: the fixture's zero is a
    strip between the column tops, not an empty gap. The 1.6 pH residual is
    the fixture detail the referee cannot model -- the short's post is a
    26 um tall block shorted to the ground plane where the DUT has only a
    2 um strip -- and it is carried as the uncertainty on the bridge
    correction."""
    with enable_x64():
        st = _study()
        xo, xi, _ = st.terminal_points(st.THETA0)
        l_bar = xo - xi
        assert abs(l_bar - 2.0 * (st.WIDTH + st.SPACING)) <= 1e-15     # 40 um, measured
        m = st.build_bar_level(1.0, l_bar)
        assert m.n_unknowns <= 25000, m.n_unknowns
        t0 = time.time()
        res = sm.solve_spiral(m, st.FREQ, sigma_volumetric=st.SIGMA_VOL)
        jax.block_until_ready(res.L_diff)
        l_de, l_raw = float(res.L_diff), float(res.L_raw)
        ref_bar = st.bar_referee(_referee(), l_bar)
        _CACHE["thru"] = {"L_raw": l_raw, "L_deembedded": l_de, "referee_bar": ref_bar,
                          "seconds": time.time() - t0}
        assert l_raw > 0 and ref_bar > 0
        # the discriminating assertion: NOT the referee bar
        assert abs(l_de) <= 0.15 * ref_bar, (l_de, ref_bar)           # measured 0.0960
        assert abs(l_de) <= 0.20 * l_raw, (l_de, l_raw)               # measured 0.0665
        # ... and the raw value IS of the size the referee predicts, so the
        # fixture is not simply broken (raw = bar + the two lead columns)
        assert 1.2 < l_raw / ref_bar < 1.8, (l_raw, ref_bar)          # measured 1.442


# ----------------------------------------------------------------------------
# T5: derivatives

def test_shape_gradients_match_finite_differences_and_the_referee_signs():
    """T5 (gates V2 / V3 in the cheap form the 100 s budget allows). The
    study checks ``jax.grad`` against FD4 with 1 % steps on every study
    grid (worst 1.60e-7 relative at W/1, 5.72e-7 at W/2, 9.28e-8 at W/3 --
    see V2 in the JSON); here the same check runs against FD2 on the
    cheaper pad = 2 grid of the same fixture (N = 11914, ~4 s per
    three-fixture solve) -- 1 % steps move L by ~1e-2 relative, far above
    the ~1e-9 LU noise floor, and the residual is then the FD2 truncation,
    measured 3.05e-5 / 1.42e-6 / 7.89e-6 relative for r_out / spacing /
    width. Gate 1e-3: 30x above the worst measurement (FD4 on the same
    grid would be ~1e-7, which is what the study asserts).

    V3 in the test form: all three ``jax.grad`` components have the SAME
    SIGN as the referee's own FD4 gradient of the DE-EMBEDDED quantity
    (+1.19167e-5, -1.08863e-5, -2.54659e-5 H/m for r_out / spacing /
    width; the strip alone would give -9.8337e-6 and -2.5001e-5 for the
    last two, because the short's bridge depends on the spacing and the
    width but not on r_out) and the same ordering by magnitude. The ratios
    on this cheap grid are 0.794 / 0.888 / 0.790; the study reports them
    per study level (0.918 / 0.936 / 0.880 at W/3, V3 in the JSON)."""
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
        assert max(rel) <= 1e-3, rel                                 # measured <= 3.1e-5 (FD2)
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
    The V1 verdict recorded there is FALSE -- the post-corrected
    extrapolated range [299.39, 306.70] pH is 7.9-10.1 % below the
    de-embedded referee's 333.055 pH -- and this test asserts that it is
    recorded as a failure with those numbers, not silently passed or
    widened. V1c (the same with the vertical grid extrapolated too,
    [307.47, 316.19] pH = -7.7 to -5.1 %) is asserted to fail as well."""
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
        # the recorded derived blocks must be exactly what the study's own
        # code recomputes from the recorded raw levels -- this is what stops
        # a gate verdict or a Richardson estimate being hand-edited
        _same(st.richardson([study["levels"][k]["base_dx"] for k in ("1", "2", "3")],
                            [study["levels"][k]["L_dut"] for k in ("1", "2", "3")]),
              study["richardson"])
        _same(st.vertical_budget(study), study["vertical_budget"])
        _same(st.evaluate_gates(study), study["gates"])
        # the de-embedded referee block reproduces from the referee alone
        dee = study["referee"]["deembedded"]
        assert abs(dee["total"] / st.referee_deembedded(_referee(), st.THETA0) - 1.0) <= 1e-12
        assert abs(dee["total"] - (dee["strip"] - dee["bridge"])) <= 1e-15 * dee["strip"]
        g = study["gates"]
        assert g["C_area"]["passed"] and g["V2"]["passed"] and g["V5"]["passed"]
        assert g["V6"]["passed"], g["V6"]
        # V6: the thru fixture de-embeds to a fraction of its own bar at every
        # level (measured 9.6 / 27.1 / 35.8 % -- NOT the 100 % that "the
        # de-embedded conductor is bar_length long" would give), and the
        # residual grows with the short's post width (10 / 20 / 23.3 um)
        v6 = g["V6"]
        res = [v6["per_level"][k] for k in ("1", "2", "3")]
        pw = [v6["post_width"][k] for k in ("1", "2", "3")]
        assert all(0.0 < r < 0.5 for r in res), res
        assert res[0] < res[1] < res[2], res
        assert pw[0] < pw[1] < pw[2], pw
        assert abs(pw[0] - 1e-5) <= 1e-12 and abs(pw[1] - 2e-5) <= 1e-12, pw
        if "thru" in _CACHE:      # T4b ran first (file order); cross-check its record
            assert abs(study["thru_probe"]["levels"]["1"]["L_deembedded"]
                       / _CACHE["thru"]["L_deembedded"] - 1.0) <= 1e-7
        assert g["V2"]["worst"] <= 1e-4, g["V2"]["worst"]
        assert abs(g["V5"]["rel_change_plus_4_cells"]) <= 0.01
        # V1 is a recorded FAILURE with numbers, not an xfail
        v1 = g["V1"]
        assert v1["passed"] is False
        assert v1["post_corrected"] is True, v1
        assert max(abs(r) for r in v1["rel_range"]) > 0.05, v1["rel_range"]
        assert -0.115 < min(v1["rel_range"]) < -0.090, v1["rel_range"]   # measured -0.1011
        assert -0.090 < max(v1["rel_range"]) < -0.065, v1["rel_range"]   # measured -0.0791
        assert 1.0 < v1["observed_order"] < 2.5, v1["observed_order"]    # measured 1.705
        # the post correction has the sign the mechanism requires (it takes
        # inductance OFF the FDFD side) at every level, and it grows
        bias = [v1["post_bias"][k] for k in ("1", "2", "3")]
        assert all(b > 0 for b in bias), bias                  # measured 1.62/4.59/6.05 pH
        assert bias[0] < bias[1] < bias[2], bias
        for k in ("W/1", "W/2", "W/3"):
            assert v1["per_level_gap"][k] < v1["per_level_gap_uncorrected"][k], (k, v1)
            assert v1["per_level_gap_vs_strip_only"][k] < v1["per_level_gap_uncorrected"][k]
        # V1c: even with the vertical direction extrapolated too it fails
        v1c = g["V1c"]
        assert v1c["passed"] is False, v1c
        assert 0.02 < min(v1c["vertical_correction_range"]) < 0.05, v1c   # measured +2.70 %
        assert max(abs(r) for r in v1c["rel_range"]) > 0.05, v1c["rel_range"]  # measured -7.7 %
        # V6's diagnosis: the residual tracks the short's post width
        assert g["V6"]["residual_tracks_post_width"] is True, g["V6"]
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
                print(f"  gap at W/1 to the de-embedded referee: {100 * _CACHE['gap']:+.3f} % "
                      f"(V1 fails); to the strip alone "
                      f"{100 * _CACHE['gap_strip_only']:+.3f} %")
            if "thru" in _CACHE:
                t = _CACHE["thru"]
                print(f"  thru probe (V6): L_raw = {t['L_raw'] * 1e12:.3f} pH, de-embedded "
                      f"{t['L_deembedded'] * 1e12:+.3f} pH = "
                      f"{100 * t['L_deembedded'] / t['referee_bar']:+.1f} % of the referee bar "
                      f"{t['referee_bar'] * 1e12:.3f} pH ({t['seconds']:.1f} s)")
            if "wall_step" in _CACHE:
                print(f"  wall gate (pad 3 -> 4): {100 * _CACHE['wall_step']:+.3f} %")
            if "grad_pad2" in _CACHE:
                g = _CACHE["grad_pad2"]
                print(f"  grad vs FD2 (pad 2): {['%.2e' % r for r in g['fd2_rel']]}, "
                      f"ratio to the referee {['%.3f' % r for r in g['ratio']]}")
        for key in ("model0", "model2", f"model{st.PAD_CELLS}"):
            if key in _CACHE:
                assert _CACHE[key].n_unknowns <= 25000, (key, _CACHE[key].n_unknowns)
