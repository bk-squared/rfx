"""Gates for the D3 design study ``validation/fdfd/spiral_design.py``: the
first gradient-based design loop on the 3-D FDFD spiral inductor, against a
3x3x3 parameter sweep over the same box.

What is and is not claimed. The study optimises a model whose ABSOLUTE
inductance is biased: D2 (``validation/fdfd/spiral_convergence.py``)
measured this solver 24.9 / 16.1 / 14.0 % BELOW an independent Greenhouse
referee at W/1 / W/2 / W/3 and its gate V1 FAILED, and D3 repeats that
comparison at its own nominal geometry (``referee_bias`` in the JSON).
What D2 validated is the SHAPE DERIVATIVE (V3: within 5-15 % of the
referee's own FD4 derivative; V2: ``jax.grad`` vs FD4 to 6e-7), and that is
what drives the loop. So the gates below are about SOLVE COUNT and
CONSTRAINT ACCURACY, never about absolute L or Q; the Leontovich sheet is
on top of that MARGINAL at this frequency (skin depth / metal thickness =
0.938 for sigma = 3e7 S/m at 2.4 GHz, the number
``spiral.leontovich_validity`` returns and the study records).

The study itself is ~1 hour of solves at N = 33352 (one three-fixture
forward solve 40-60 s under contention), so it cannot be re-run here. The
gates are asserted from its JSON, which is first checked for INTERNAL
consistency against its own raw per-evaluation log (the loss, the L error,
lambda and the gate booleans are all recomputed here from the logged L, Q
and gradients), and the loop DRIVER is then re-run live on a cheap model.

D1  the JSON exists, matches the study module's constants, and every
    derived number in it is reproduced from the raw log: loss from (L, Q),
    the L errors, lambda from the two measured gradients, and the whole
    gate dictionary.
D2  O1: the loss decreases over the accepted iterates.
D3  O2: final |L - L_target| / L_target <= 2 %, and lambda's design bound
    (1 %) is the reason -- the achieved error is compared against it.
D4  O3: the final Q against the best sweep Q inside the 2 % L band, with
    the "no sweep point qualifies" branch asserted as a fact about the
    sweep (how many of the 27 forward solves landed in the band) rather
    than waived.
D5  O4: ``jax.grad`` of the FULL objective at the final theta vs FD4 with
    1 % scaled steps, from the study's 12 forward solves.
D6  O5 (report only): the W/2 re-solve. No pass/fail on the SIZE of the
    shift and no window around D2's +11.66 % -- D2 measured that on a
    different geometry, so only its SIGN transfers. What is asserted is
    the arithmetic, the sign (both points move UP), and the claim the
    block exists to make: the shift is COMMON MODE (measured
    ``common_mode_fraction`` = 0.950, asserted > 0.5).
D8  M1: the memory rule, and it FAILS. Every block records the physical
    footprint it reached and the limit it ran under. Three controlled
    series (same theta, one process, cache cleared, gc forced) separate
    what grows from what does not: ``solve_spiral`` un-jitted is flat
    (4.94 -> 4.79 GB over five solves), the SAME solve jitted grows
    +3.351 GB per solve (5.03 -> 15.08 over four) and one
    ``value_and_grad`` pass +3.904 GB. Hence one loop evaluation per
    process (16 chunks, 4.60-5.29 GB, all inside 6 GB) -- and hence the
    two blocks measured before that instrumentation existed broke the
    rule: sweep 8.39 GB, FD stencil 21.64 GB. The test asserts the
    failure with its numbers instead of waiving it.
D7  LIVE: the same driver (``choose_lambda`` -> ``run_loop``) on the cheap
    default ``SpiralSpec`` (N = 12739, one forward solve ~5 s), 2 iterates,
    with a 4-point central difference of the objective along ``width``
    against ``jax.grad``. This is the part that would catch a broken
    driver; the expensive study only supplies the numbers.

The one check that is NOT here is the drift check on the recorded numbers
themselves -- "does the stored nominal still come out of the current
``rfx.fdfd.spiral``" costs one 40 s solve and would put this file over its
100 s budget, so it is script-only:
``validation/fdfd/spiral_design.py --verify`` (run against the concurrently
edited ``spiral.py`` in this session: bit-identical, 0.0e0 relative).

x64 is scoped per test through ``tests._x64_compat.enable_x64``. Measured
wall time of this file in this session: see the module-level comment at
``_SMOKE_SECONDS``.
"""
from __future__ import annotations

import importlib.util
import json
import math
import pathlib
import time

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx.fdfd import spiral as sm
from tests._x64_compat import enable_x64

pytest.importorskip("shapely")

REPO = pathlib.Path(__file__).resolve().parents[3]
STUDY_PATH = REPO / "validation" / "fdfd" / "spiral_design.py"
JSON_PATH = REPO / "validation" / "fdfd" / "spiral_design.json"
PNG_PATH = REPO / "validation" / "fdfd" / "spiral_design.png"
CONV_JSON = REPO / "validation" / "fdfd" / "spiral_convergence.json"

# Measured in this session: the whole file 56.7 s, of which 55.4 s is the
# live smoke test D7 (build, one nominal forward solve, choose_lambda = 1
# forward + 2 adjoints, 2 L-BFGS-B iterates over 4 evaluations, and a
# 4-point FD4 stencil, all on the N = 12739 model). The seven JSON gates
# are pure arithmetic: 0.77 s together.
_SMOKE_SECONDS = 55.4

_CACHE: dict = {}


def _study():
    """The study module (it owns the problem constants and the driver; the
    test imports it so the two cannot drift apart)."""
    if "study" not in _CACHE:
        spec = importlib.util.spec_from_file_location("spiral_design_study", STUDY_PATH)
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _CACHE["study"] = mod
    return _CACHE["study"]


def _json() -> dict:
    if "json" not in _CACHE:
        if not JSON_PATH.exists():
            pytest.skip(f"{JSON_PATH} missing: run validation/fdfd/spiral_design.py")
        _CACHE["json"] = json.loads(JSON_PATH.read_text())
    return _CACHE["json"]


# ----------------------------------------------------------------------------
# D1: the JSON is self-consistent

def test_the_study_json_is_internally_consistent_with_its_own_raw_log():
    """D1. Everything derived in the JSON is recomputed here from the raw
    per-evaluation log, so a hand-edited summary cannot pass the gates:

    * the fixture block equals the study module's constants (frequency,
      conductivities, nominal theta, box, pad cells, real permittivities);
    * ``L_target = 0.85 x`` the nominal ``L_diff`` and ``Q_ref`` is the
      nominal ``Q_diff`` (to 1e-15 relative);
    * ``lam = |g_Q| / (2 e_target |g_L|)`` from the two gradients the study
      measured by reverse mode at the nominal theta (1e-12);
    * every logged ``loss`` equals ``-Q/Q_ref + lam ((L - L_t)/L_t)^2``
      recomputed from the logged L and Q (1e-12 relative), and every
      logged ``L_rel_error`` equals ``L / L_t - 1`` (1e-12);
    * the sweep's ``best_in_band`` really is the highest-Q point inside the
      band, and the counts add up;
    * ``evaluate_gates`` re-run on the loaded JSON reproduces the recorded
      gate dictionary.
    """
    st, d = _study(), _json()
    fx = d["fixture"]
    assert fx["freq"] == st.FREQ and fx["sigma_metal"] == st.SIGMA_CU
    assert fx["sigma_si"] == st.SIGMA_SI and fx["n_turns"] == st.N_TURNS
    assert fx["theta_nominal"] == list(st.THETA0)
    assert [tuple(b) for b in fx["bounds"]] == [tuple(b) for b in st.BOUNDS]
    assert fx["pad_cells"] == st.PAD_CELLS == 4
    assert (fx["eps_si"], fx["eps_ox"]) == (st.EPS_SI, st.EPS_OX) == (11.9, 4.1)
    assert fx["eps_si"] > 1.0 and fx["eps_ox"] > 1.0      # the REAL stack, not D2's vacuum

    nom, des, lam_rec, loop = d["nominal"], d["design"], d["lambda"], d["loop"]
    assert abs(des["L_target"] - st.L_FRACTION * nom["L_diff"]) <= 1e-15 * des["L_target"]
    assert des["Q_ref"] == nom["Q_diff"]
    n_q = float(np.linalg.norm(lam_rec["g_Q_over_Qref"]))
    n_l = float(np.linalg.norm(lam_rec["g_L_over_Lt"]))
    lam = n_q / (2.0 * st.E_TARGET * n_l)
    assert abs(lam - lam_rec["lam"]) <= 1e-12 * lam
    assert abs(lam - loop["lam"]) <= 1e-12 * lam

    l_t, q_ref = loop["L_target"], loop["Q_ref"]
    for e in loop["evals"]:
        loss = st.loss_of(e["L"], e["Q"], l_t, q_ref, lam)
        assert abs(loss - e["loss"]) <= 1e-12 * max(abs(loss), 1.0), e["i"]
        assert abs((e["L"] / l_t - 1.0) - e["L_rel_error"]) <= 1e-12
        assert e["feasibility_margin"] > 0.0, e            # the box stayed topologically valid
    assert loop["n_forward_solves"] == loop["n_adjoint_passes"] == len(loop["evals"])
    assert loop["n_distinct_thetas"] + loop["n_cached_repeats"] == len(loop["evals"])

    sw = d["sweep"]
    ok = [p for p in sw["points"] if p["feasible"]]
    assert len(sw["points"]) == st.SWEEP_N ** 3 == 27
    assert len(ok) == sw["n_forward_solves"]
    inside = [p for p in ok if abs(p["L_rel_error"]) <= st.L_BAND]
    assert len(inside) == sw[f"n_in_band_{st.L_BAND:g}"]
    best = sw[f"best_in_band_{st.L_BAND:g}"]
    if inside:
        assert best is not None and best["Q"] == max(p["Q"] for p in inside)
    else:
        assert best is None

    assert st.evaluate_gates(d) == d["gates"]
    assert PNG_PATH.exists() and PNG_PATH.stat().st_size > 20000


# ----------------------------------------------------------------------------
# D2: O1

def test_gate_O1_the_loss_decreases_over_the_accepted_iterates():
    """O1. The accepted iterates (the points L-BFGS-B's callback reported,
    as opposed to the line-search trials, which the study logs separately)
    must have a monotonically decreasing loss. The gate is the worst
    increase between consecutive accepted losses, which must be <= 0.
    Measured: 12 accepted iterates over 16 objective evaluations, loss
    -0.649875 -> -1.062610, worst increase -6.58e-5.

    The stop criterion is also checked here, because the naive reading of
    the log is wrong: L-BFGS-B stopped on "NORM OF PROJECTED GRADIENT <=
    PGTOL" with a projected sup-norm of 8.61e-5 against the gtol of
    7.28e-3 (1e-3 x its value at the nominal theta), while the FULL
    gradient at that same point is 0.399 -- the optimum sits on the
    ``spacing`` lower bound and the ``width`` upper bound, where the
    outward components are projected away."""
    d = _json()
    g = d["gates"]["O1"]
    acc = [e for e in d["loop"]["evals"] if e["accepted"]]
    assert len(acc) >= 3, len(acc)
    losses = [e["loss"] for e in acc]
    assert losses == g["losses"]
    rises = [losses[i + 1] - losses[i] for i in range(len(losses) - 1)]
    assert max(rises) <= 0.0, rises
    assert g["passed"] is True
    assert losses[-1] < losses[0]                 # and it actually moved

    # the stop criterion: scipy's gtol tests the PROJECTED gradient, and the
    # optimum of this problem sits on box bounds, so the full |grad| logged
    # per evaluation stays O(1) while the projected one collapses. Both are
    # recomputed here from the raw log.
    st = _study()
    stop = d["gates"]["stopping"]
    bnds = d["loop"]["bounds_scaled"]
    pg = [float(np.max(np.abs(st.projected_gradient(e["u"], e["grad"], bnds)))) for e in acc]
    assert pg == pytest.approx(stop["projected_grad_inf_per_accepted_iterate"], rel=1e-12)
    assert pg[-1] < pg[0], (pg[0], pg[-1])
    assert stop["full_grad_inf_final"] == acc[-1]["grad_inf"]
    assert any(stop["active_bounds_final"].values())      # the optimum is on a bound


# ----------------------------------------------------------------------------
# D3: O2

def test_gate_O2_the_inductance_constraint_binds_to_the_designed_accuracy():
    """O2. ``|L - L_target| / L_target <= 2 %`` at the final theta, where
    ``L_target = 0.85 x`` the nominal L. The penalty weight was not tuned to
    make this pass: ``lam = |g_Q| / (2 e_target |g_L|)`` with
    ``e_target = 1 %`` is the bound on the relative L error at an interior
    stationary point (``-g_Q + 2 lam e g_L = 0``), computed from two
    reverse-mode gradients at the nominal theta. This test asserts the 2 %
    gate AND that the achieved error respects the 1 % design bound the
    weight was sized for -- i.e. that the bound was predictive."""
    d = _json()
    loop, g = d["loop"], d["gates"]["O2"]
    e = abs(loop["final_L_rel_error"])
    assert e == pytest.approx(g["measured"], rel=1e-12)
    assert e <= 0.02, e
    # lambda's bound is |e| <= e_target = 1 %; the loop achieved 1.019 %
    # (measured). The bound is not an identity here for two reasons, both
    # stated rather than tuned away: |g_Q| and |g_L| are measured at the
    # NOMINAL theta, not at the optimum, and the optimum sits on two active
    # box bounds, so the interior stationarity argument only holds in the
    # free subspace. The gate is therefore "the bound was predictive to
    # better than 50 %", measured 1.9 % relative (1.019 % vs 1 %).
    assert e <= 1.5 * d["fixture"]["e_target"], (e, d["fixture"]["e_target"])
    assert g["passed"] is True
    # the loop actually moved L toward the target: the nominal is 1/0.85 - 1
    # = +17.6 % away, the final is inside the band
    assert abs(d["nominal"]["L_diff"] / loop["L_target"] - 1.0) > 0.15


# ----------------------------------------------------------------------------
# D4: O3

def test_gate_O3_the_gradient_optimum_against_the_sweep_baseline():
    """O3. The final Q against the best Q of the 27-point sweep INSIDE the
    2 % L band. Two branches, both asserted, neither waived:

    (a) if any sweep point is in the band, the gradient optimum must have
        the higher Q;
    (b) if none is -- the expected outcome of a blind 3x3x3 grid against a
        band that is 2 % wide in a quantity that varies by tens of percent
        over the box -- the gate reports that, and the test asserts the
        FACT behind it (the count of in-band sweep points is 0 and the
        widest band that catches any sweep point is recorded) plus the
        weaker but still meaningful comparison that the gradient point has
        the lower objective value than every sweep point.

    Measured: branch (b). All 27 sweep points are feasible, NONE is inside
    the 2 % band (4 inside 5 %, 7 inside 10 %, 17 inside 25 %), the best
    sweep loss is -1.053077 at (72, 10, 12) um with Q = 11.7502 at L
    +2.93 %, and the gradient optimum is -1.062610 with Q = 11.7618 at L
    +1.019 %. Cost, summed from the per-solve seconds in the raw logs:
    the sweep spent 27 forward solves and 1053.6 s, the loop 16 forward +
    16 adjoint passes and 450.5 s, and the FD4 gradient at ONE point (gate
    O4) 12 forward solves and 493.1 s -- more than the whole loop."""
    d = _json()
    g, loop, sw = d["gates"]["O3"], d["loop"], d["sweep"]
    ok = [p for p in sw["points"] if p["feasible"]]
    assert len(ok) >= 20                       # the box is feasible almost everywhere
    inside = [p for p in ok if abs(p["L_rel_error"]) <= 0.02]
    if inside:
        assert g["passed"] is True
        assert loop["final_Q"] >= g["best_sweep_Q"], (loop["final_Q"], g["best_sweep_Q"])
    else:
        assert g["passed"] is None and g["sweep_points_in_band"] == 0
        assert "NO sweep point qualifies" in g["statement"]
        assert loop["final_loss"] < min(p["loss"] for p in ok), (
            loop["final_loss"], min(p["loss"] for p in ok))
    # the sweep cost 27 forward solves and produced no gradient information;
    # the loop's cost is its own evaluation count (1 forward + 1 adjoint each)
    assert sw["n_forward_solves"] + sw["n_infeasible"] == 27
    # the cost comparison, recomputed from the raw per-solve seconds
    assert g["loop_evaluations"] == len(d["loop"]["evals"]) == 16
    assert g["loop_solve_seconds"] == pytest.approx(
        sum(e["seconds"] for e in d["loop"]["evals"]), rel=1e-12)
    assert g["sweep_solve_seconds"] == pytest.approx(
        sum(pt["seconds"] for pt in sw["points"]), rel=1e-12)
    assert g["sweep_solve_seconds"] > 2.0 * g["loop_solve_seconds"]   # 1053.6 vs 450.5
    assert g["fd4_gradient_forward_solves"] == 12


# ----------------------------------------------------------------------------
# D5: O4

def test_gate_O4_jax_grad_of_the_objective_matches_fd4_at_the_optimum():
    """O4. ``jax.grad`` of the FULL objective (not just L) at the final
    theta against a 4th-order central difference with 1 % steps on the
    scaled parameters -- the same step D2 used, far above the ~1e-9
    relative LU noise floor.

    The METRIC needs care and the study records both forms. At a
    CONSTRAINED optimum the free-direction component of the gradient is
    zero by stationarity: measured ``dloss/dr_out = 8.613e-5`` against a
    gradient sup norm of 0.3994, i.e. 2.2e-4 of it. Dividing that
    component's absolute FD-truncation error (3.02e-7, the same size as the
    other two components' 6.1e-9 and 4.3e-8) by the component itself gives
    a "relative disagreement" of 3.5e-3 while the two bound-active
    components agree to 3.4e-7 and 1.1e-7. The number that means something
    is the error relative to the gradient NORM -- what an optimiser
    consumes -- measured 7.55e-7, and that is what the 1e-4 gate is stated
    on. Both are asserted here: the norm-relative error against 1e-4, and
    the two components that are NOT zero against 1e-4 per component.

    The same 12 forward solves also give FD4 of L and of Q separately, and
    those are checked against the objective's FD4 through the chain rule
    ``dloss = -dQ/Q_ref + 2 lam e dL/L_t`` (worst absolute difference
    measured 3.55e-7 = 8.9e-7 of the gradient norm, gate 1e-5) -- which is
    what makes this a check of the derivative of the OBJECTIVE and not only
    of L."""
    d = _json()
    fd, g = d["fd_check"], d["gates"]["O4"]
    assert fd["fd_order"] == 4 and fd["step_rel"] == 0.01 and fd["complete"]
    grad = np.asarray(fd["grad"], dtype=np.float64)
    fd4 = np.asarray(fd["fd4"], dtype=np.float64)
    err = np.abs(fd4 - grad)
    norm = float(np.max(np.abs(grad)))
    assert err.tolist() == pytest.approx(g["abs_err"], rel=1e-12)
    assert float(np.max(err)) / norm == pytest.approx(g["rel_to_grad_norm"], rel=1e-12)
    assert g["rel_to_grad_norm"] <= 1e-4, g["rel_to_grad_norm"]      # measured 7.55e-7
    assert g["passed"] is True
    # the components that are not zero at the optimum also pass the naive
    # per-component form; the one that fails it is the one that is zero
    big = np.abs(grad) / norm > 1e-3
    assert big.sum() == 2, grad.tolist()
    assert max(np.asarray(g["rel_per_component"])[big]) <= 1e-4
    assert g["smallest_component_over_norm"] < 1e-3                  # measured 2.16e-4

    lam, l_t, q_ref = d["loop"]["lam"], d["loop"]["L_target"], d["loop"]["Q_ref"]
    e = fd["L"] / l_t - 1.0
    for k in range(3):
        chain = -fd["fd4_Q"][k] / q_ref + 2.0 * lam * e * fd["fd4_L"][k] / l_t
        assert abs(chain - fd["fd4"][k]) <= 1e-5 * norm, (k, chain, fd["fd4"][k])
    assert abs(fd["fd4_L"][2]) > 0 and abs(fd["fd4_Q"][2]) > 0       # both terms are live


# ----------------------------------------------------------------------------
# D6: O5 (report only)

def test_gate_O5_the_finer_grid_resolve_is_reported_and_is_mostly_common_mode():
    """O5 is REPORT ONLY: the final theta re-solved at ``base_dx = W/2``.
    A pass/fail on the size of the shift would be meaningless -- the coarse
    model is known to be 15-25 % low on absolute L (D2's V1 failed and D3's
    own ``referee_bias`` block repeats that comparison at this geometry),
    so a shift of that order is exactly what must happen.

    What is asserted are the three statements that do NOT need a tolerance:

    * arithmetic -- the recorded shifts are recomputed from the recorded L;
    * SIGN -- both points move UP, which is the one thing D2's per-level
      gaps (-24.9 % at W/1, -16.1 % at W/2, -14.0 % at W/3, monotone)
      predict for a different geometry. D2's implied ``+11.66 %`` is
      recorded as CONTEXT and deliberately not used as a window: an earlier
      revision of this test required the shift to be within a factor of two
      of it (+5.9 .. +23.4 %), which is a judgment and not a measurement --
      the two fixtures differ in ``r_out`` (72 vs 52 um), in the
      dielectrics (real vs vacuum) and in the metal model (Leontovich vs
      uniform-current), so nothing measured licenses that window;
    * COMMON MODE -- the claim this block actually makes. The nominal and
      the optimum move by nearly the same amount, so the part of the shift
      that could move the DESIGN is small: measured +6.810 % (nominal) vs
      +6.468 % (optimum), a differential of -0.341 points on a 6.810-point
      move, i.e. ``common_mode_fraction`` = 0.950. The assertion is the
      qualitative claim -- differential smaller than half the common part
      (fraction > 0.5) -- which the measurement clears by 10x (the
      differential is 5.0 % of the common shift), not a band fitted to it.

    Also asserted: the L error against the W/1 target GROWS (+1.019 % ->
    +7.55 %, the honest consequence -- a target hit to 1 % on the coarse
    grid is not hit on the finer one) and the finer grid really is finer.
    """
    d = _json()
    g, ref = d["gates"]["O5"], d["refine"]
    assert g["passed"] is None                       # report only, by construction
    conv = json.loads(CONV_JSON.read_text())
    gaps = [conv["levels"]["1"]["L_dut"], conv["levels"]["2"]["L_dut"]]
    d2_implied = gaps[1] / gaps[0] - 1.0             # +11.66 %, measured by D2
    assert g["d2_implied_shift_W1_to_W2"] == pytest.approx(d2_implied, rel=1e-12)
    for name in ("nominal", "final"):
        p = ref["points"][name]
        assert p["L_shift"] == pytest.approx(p["L"] / p["L_coarse"] - 1.0, rel=1e-12)
        assert p["L_shift"] > 0.0, (name, p["L_shift"])          # the SIGN D2 predicts
        assert math.isfinite(p["Q_shift"])
    assert g["same_sign_as_d2"] is True
    diff = ref["points"]["final"]["L_shift"] - ref["points"]["nominal"]["L_shift"]
    big = max(abs(ref["points"]["final"]["L_shift"]), abs(ref["points"]["nominal"]["L_shift"]))
    assert g["L_shift_differential"] == pytest.approx(diff, rel=1e-12)
    assert g["common_mode_fraction"] == pytest.approx(1.0 - abs(diff) / big, rel=1e-12)
    assert g["common_mode_fraction"] > 0.5, g["common_mode_fraction"]   # measured 0.950
    fine = ref["points"]["final"]["L_rel_error"]
    assert abs(fine) > abs(d["loop"]["final_L_rel_error"]), fine
    assert ref["grid"]["n_unknowns"] > d["nominal"]["grid"]["n_unknowns"]


# ----------------------------------------------------------------------------
# D8: the peak-RSS rule (gate M1)

def test_gate_M1_memory_FAILED_on_two_blocks_and_the_growth_is_in_the_jitted_path():
    """M1. This machine runs four agents on 38 GB and the rule for this
    track is ~6 GB of peak memory per solve process, so the study MEASURES
    it -- and the first thing that measurement had to get right is WHICH
    number.

    ``ru_maxrss`` and ``ps -o rss=`` understate these processes, worst when
    the machine is worst off: the pages a solve leaves behind go cold, and
    cold pages are what macOS compresses and swaps first. Measured on a
    loop process that had run 12 evaluations: ru_maxrss 5.93 GB, resident
    0.007 GB, actual physical footprint ~16 GB (killing it returned 16 GB
    of swap on a machine that was 37.8 of 38.9 GB into swap at load 144).
    So the study records ``ri_phys_footprint`` from ``proc_pid_rusage``
    (verified against ``vmmap --summary`` to 0.03 GB) after every solve,
    per block and per evaluation. That instrumentation replaced two
    contradictory RSS claims in an earlier revision (+3.6 GB per solve in
    the module docstring, ~0.5 GB per solve in the budget comment).

    WHAT GROWS, measured as three controlled series in this session -- same
    theta, one process, N = 33352, ``clear_factor_cache()`` and
    ``gc.collect()`` between solves (``memory.probe``, ``probe_jit``,
    ``probe_grad``), footprint in GB after 1, 2, ... solves:

        solve_spiral, NO jit      4.94  4.96  4.69  4.70  4.79
        the same solve JITTED     5.03  8.37 11.71 15.08
        jax.value_and_grad of it  4.60  8.51  (stopped: over the limit)

    i.e. the un-jitted path is FLAT (-0.052 GB/solve, fit residual 0.124
    GB) and the JITTED one -- the path the loop, the sweep and the FD
    stencil all take -- grows +3.351 GB per forward solve (residual 0.007
    GB over four points; resident +2.495) and +3.904 GB per reverse pass.
    The cause is below ``rfx.fdfd.linear_solve``'s ``pure_callback`` and is
    NOT diagnosed here.

    The consequences this test asserts:

    * the loop gets ONE evaluation per process and resumes by replaying its
      log: 16 chunks of one solve, footprints 4.60-5.29 GB, all inside 6.
      The same 16 in one process would have reached 4.60 + 15 x 3.904 =
      63 GB, which this machine does not have;
    * the gate FAILS on the two blocks that were measured before the
      instrumentation existed and did put several jitted solves in one
      process: the sweep reached 8.39 GB and the FD stencil 21.64 GB
      against the 6 GB they ran under. Reported failing, not widened and
      not xfailed; re-running them one solve per process costs 39 solves
      (~26 min) and buys no new physics;
    * three blocks ran under an explicitly RAISED limit, passed on the
      command line and recorded per block: factorisations 8.38 (12),
      probe_jit 15.08 (16), the W/2 re-solve 18.21 (22). The reverse-mode
      probe is over the limit BY DESIGN (it measures the crossing) and is
      listed separately from the two failures;
    * every block that ran is measured; the one block NOT RUN
      (``mode_cost``, 20-25 GB) says why in the JSON.
    """
    d = _json()
    st = _study()
    g, mem = d["gates"]["M1"], d["memory"]
    blocks = mem["by_block"]
    assert set(blocks) >= {"nominal", "lambda", "loop", "sweep", "fd_check",
                           "factorisations", "referee_bias", "refine"}, sorted(blocks)
    for name, rec in blocks.items():
        peak = g["footprint_gb_by_block"][name]
        assert peak >= rec.get("footprint_gb_max", rec["footprint_gb_after"]) - 1e-12
        assert peak > 0.5                                   # it really was measured
        assert rec["limit_gb"] >= st.MEM_LIMIT_GB == 6.0    # a limit is never LOWERED
        assert g["limit_gb_by_block"][name] == rec["limit_gb"]
    for name in ("nominal", "lambda", "loop", "sweep", "fd_check", "referee_bias"):
        assert blocks[name]["limit_gb"] == st.MEM_LIMIT_GB, name
    raised = {n for n, b in blocks.items() if b["limit_gb"] > st.MEM_LIMIT_GB}
    assert raised == {"factorisations", "memory_probe_jit", "refine"}, raised

    # the failure, asserted as a fact with its numbers (never waived)
    assert g["passed"] is False
    assert set(g["blocks_over_limit"]) == {"sweep", "fd_check"}, g["blocks_over_limit"]
    assert g["blocks_over_limit"]["sweep"][0] == pytest.approx(8.391, rel=1e-3)
    assert g["blocks_over_limit"]["fd_check"][0] == pytest.approx(21.641, rel=1e-3)
    for over, limit in g["blocks_over_limit"].values():
        assert over > limit == st.MEM_LIMIT_GB
    assert set(g["blocks_over_limit_by_design"]) == {"memory_probe_grad"}
    # nothing measured is missing, and the block that was not RUN says why
    assert g["blocks_not_measured"] == [] and mem["not_measured"] == {}
    assert set(d["not_run"]) == set(st.BLOCKS_NOT_RUN) == {"mode_cost"}
    assert len(d["not_run"]["mode_cost"]) > 200             # the reason, not a shrug

    # ---- the three series -------------------------------------------------
    pr, pj, pg = mem["probe"], mem["probe_jit"], mem["probe_grad"]
    for rec in (pr, pj, pg):
        assert rec["clear_cache"] is True and rec["L_spread"] == 0.0
        assert rec["grid"]["n_unknowns"] == d["nominal"]["grid"]["n_unknowns"] == 33352
        assert rec["theta"] == d["fixture"]["theta_nominal"]
        assert len(rec["footprint_gb"]) == rec["n_solves"] >= 2
    assert (pr["mode"], pj["mode"], pg["mode"]) == ("forward", "jit", "grad")
    assert pr["n_solves"] >= 4 and pj["n_solves"] >= 4      # RSS after 1..4 solves

    # the un-jitted path does not grow. Threshold from the measurement: the
    # slope is -0.052 GB/solve with a 0.124 GB fit residual, i.e. flat to the
    # noise of the series, so anything under half a GB per solve is "flat"
    # and the jitted +3.351 is 64x outside it.
    assert abs(pr["growth_gb_per_solve"]) < 0.5, pr["growth_gb_per_solve"]
    assert pr["footprint_gb_max"] < st.MEM_LIMIT_GB and pr["stopped_at_limit"] is False
    assert max(pr["footprint_gb"]) - min(pr["footprint_gb"]) < 0.5   # measured 0.28

    # the jitted path does, and linearly: 5.03 -> 15.08 GB over four solves
    assert pj["growth_gb_per_solve"] > 2.0, pj["growth_gb_per_solve"]   # measured 3.351
    assert pj["growth_fit_residual_gb"] < 0.1                          # measured 0.0066
    assert pj["growth_gb_per_solve"] > 5.0 * abs(pr["growth_gb_per_solve"])
    # at the 6 GB rule that slope gives ONE jitted solve per process
    # (recomputed from the raw series by evaluate_gates, because this probe
    # itself had to run under a raised limit to reach four solves)
    assert g["probe_jit_solves_per_process_at_default_limit"] == 1
    assert g["probe_jit_limit_gb"] > st.MEM_LIMIT_GB
    fps = pj["footprint_gb"]
    for i in range(1, len(fps)):                            # monotone, unlike RSS
        assert fps[i] > fps[i - 1], fps

    # reverse mode is worse again, and it stops itself at the crossing
    assert pg["growth_gb_per_solve"] > pj["growth_gb_per_solve"]        # 3.904 vs 3.351
    assert pg["stopped_at_limit"] is True
    assert pg["footprint_gb"][0] <= st.MEM_LIMIT_GB < pg["footprint_gb"][-1]
    assert pg["solves_per_process_at_limit"] == 1           # hence one evaluation per chunk

    # ---- the loop: one evaluation per process, and the peak it avoided ----
    fp = [e["mem"]["footprint_gb"] for e in d["loop"]["evals"]]
    assert len(fp) == d["loop"]["n_objective_evaluations"] == 16
    assert max(fp) <= st.MEM_LIMIT_GB, max(fp)
    assert [c["solves"] for c in blocks["loop"]["chunks"]] == [1] * 16
    implied = fp[0] + (len(fp) - 1) * pg["growth_gb_per_solve"]
    assert implied > 50.0, implied                          # measured 63 GB
    assert d["loop"]["clear_cache"] is True
    assert d["loop"]["n_replay_mismatches"] == 0            # the replay was exact
    assert d["loop"]["n_evals_replayed"] >= 1               # it really was resumed


# ----------------------------------------------------------------------------
# D7: the driver, live, on a cheap model

def test_the_design_driver_runs_and_its_gradient_matches_a_central_difference():
    """D7. The expensive study supplies numbers; this runs the DRIVER.

    Same code path (``choose_lambda`` -> ``objective_fn`` -> ``run_loop``)
    on the cheap default ``SpiralSpec`` (N = 12739, pad = 0, one
    three-fixture forward solve 4.7 s measured here), 2.4 GHz, the same
    Leontovich copper and lossy silicon, a small box around its own
    nominal, ``maxiter = 2``. Asserted:

    * the loop's accepted losses decrease, and after 2 iterates the L error
      is already inside the 2 % band (measured -0.21 % in this session:
      the same convergence the full study shows, on a 2.6x cheaper model);
    * ``jax.grad`` of the objective at the final theta vs a 4-point central
      difference (FD4) in ``width`` with a 1 % step: measured 2.19e-7
      relative here, gate 1e-4. The 2-point stencil at the same step was
      measured at 1.7e-3, i.e. the O(h^2) truncation -- which is why the
      gate uses FD4 and not FD2;
    * ``check_feasible`` holds at every iterate and the LU-cache repeat of
      x0 really is nearly free (the second evaluation of the identical
      theta measured 0.1 s against 4.7 s).
    """
    with enable_x64():
        st = _study()
        t_start = time.time()
        model = sm.build_spiral(sm.SpiralSpec())
        assert model.n_unknowns == 12739, model.n_unknowns
        theta0 = (60e-6, 10e-6, 10e-6)
        bounds = ((56e-6, 66e-6), (9e-6, 11e-6), (9e-6, 11e-6))
        raw = jax.jit(st.metrics_fn(model, theta0=theta0))
        l0, q0 = raw(jnp.ones(3, dtype=jnp.float64))
        l0, q0 = float(l0), float(q0)
        assert l0 > 0 and q0 > 0
        l_t, q_ref = 0.85 * l0, q0
        lam_rec = st.choose_lambda(model, l_t, q_ref, verbose=False, theta0=theta0)
        assert lam_rec["lam"] > 0
        assert lam_rec["lam"] == pytest.approx(
            np.linalg.norm(lam_rec["g_Q_over_Qref"])
            / (2 * st.E_TARGET * np.linalg.norm(lam_rec["g_L_over_Lt"])), rel=1e-12)

        rec = st.run_loop(model, l_t, q_ref, lam_rec["lam"], maxiter=2, verbose=False,
                          theta0=theta0, bounds=bounds)
        acc = [e["loss"] for e in rec["evals"] if e["accepted"]]
        assert len(acc) >= 2
        assert all(acc[i + 1] < acc[i] for i in range(len(acc) - 1)), acc
        assert abs(rec["final_L_rel_error"]) <= 0.02, rec["final_L_rel_error"]
        assert rec["n_cached_repeats"] >= 1              # scipy re-evaluates x0
        for e in rec["evals"]:
            sm.check_feasible(model, e["theta"])

        # FD4 of the objective in width at the final theta vs jax.grad
        _, fwd = st.objective_fn(model, l_t, q_ref, lam_rec["lam"], theta0=theta0)
        u = np.asarray(rec["final_u"], dtype=np.float64)
        h = 0.01
        vals = []
        for m in (2.0, 1.0, -1.0, -2.0):
            up = u.copy()
            up[2] += m * h
            vals.append(float(fwd(jnp.asarray(up))[0]))
        p2, p1, m1, m2 = vals
        fd4 = (-p2 + 8.0 * p1 - 8.0 * m1 + m2) / (12.0 * h)
        g = rec["final"]["grad"][2]
        rel = abs(fd4 - g) / abs(g)
        assert rel <= 1e-4, (fd4, g, rel)               # measured 2.19e-7 (FD2 at the same step: 1.7e-3)
        _CACHE["smoke_seconds"] = time.time() - t_start
        assert _CACHE["smoke_seconds"] < 300.0          # contention budget
