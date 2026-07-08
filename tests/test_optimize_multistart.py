"""Contract tests for the opt-in multi-start / best-iterate / step-clamp
knobs promoted into ``rfx.optimize`` (WP 4-C, from the MSL open-stub
example's ``_multistart_adam``, issue #171).

Two layers:

1. **Bit-identity gate (load-bearing).** With ``n_starts=1``,
   ``best_iterate=False`` and ``step_clamp=None`` the promoted
   ``_adam_multistart`` core must reproduce the LEGACY inline Adam loop
   BYTE-for-BYTE — both on a synthetic cost (portable, FDTD-free) and
   end-to-end through ``optimize()`` (adding the new defaulted params
   changes nothing).  This is the non-negotiable falsifier: if it ever
   fails, the default path drifted.

2. **Six MSL-G2 behaviors** replicated at the library level against
   synthetic costs (mirrors ``tests/test_msl_multistart.py`` for the
   example), plus the generic step-clamp semantics and fail-closed guards.

NOTE: do NOT call ``jax.config.update("jax_enable_x64", True)`` at module
level — it is a PROCESS-GLOBAL flag and contaminates other pytest-split
shards.  These tests use float32 (the production dtype); tolerances are
float32-safe.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import jax
import jax.numpy as jnp

from rfx.optimize import _adam_multistart, optimize, DesignRegion, OptimizeResult


# ---------------------------------------------------------------------------
# Reference: verbatim copy of the LEGACY (pre-WP-4-C) inline optimize() Adam
# loop.  The bit-identity gate asserts the promoted core reproduces this.
# ---------------------------------------------------------------------------
def _legacy_single_start_adam(cost_fn, init_latent, n_iters, lr):
    beta1, beta2, eps_adam = 0.9, 0.999, 1e-8
    m = jnp.zeros_like(init_latent)
    v = jnp.zeros_like(init_latent)
    latent = init_latent
    loss_history = []
    grad_fn = jax.value_and_grad(cost_fn)
    last_good = latent
    for it in range(n_iters):
        loss, grad = grad_fn(latent)
        if not (bool(jnp.isfinite(loss)) and bool(jnp.all(jnp.isfinite(grad)))):
            latent = last_good
            break
        last_good = latent
        loss_val = float(loss)
        loss_history.append(loss_val)
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad ** 2
        m_hat = m / (1 - beta1 ** (it + 1))
        v_hat = v / (1 - beta2 ** (it + 1))
        latent = latent - lr * m_hat / (jnp.sqrt(v_hat) + eps_adam)
    return loss_history, latent


def _bowl_cost(z):
    """Smooth quadratic bowl over a small latent array (real Adam path)."""
    target = jnp.asarray([1.0, -1.0, 0.5], dtype=jnp.float32)
    return jnp.sum((z - target) ** 2)


# ---------------------------------------------------------------------------
# 1. BIT-IDENTITY GATE (load-bearing) — synthetic, portable
# ---------------------------------------------------------------------------
def test_bit_identity_default_matches_legacy_loop():
    """`_adam_multistart` at defaults == the legacy inline Adam loop, bit-for-bit.

    This is the committed form of the WP-4-C falsifier: the default path
    (single start, no best-iterate, no clamp) must be byte-identical to the
    algorithm that shipped before the knobs existed.
    """
    init = jnp.asarray([0.5, -0.3, 1.2], dtype=jnp.float32)
    n_iters, lr = 8, 0.1

    legacy_lh, legacy_latent = _legacy_single_start_adam(_bowl_cost, init, n_iters, lr)
    best_latent, lh, all_h, best_start = _adam_multistart(
        _bowl_cost, [init], n_iters=n_iters, lr=lr,
    )

    assert best_start == 0
    assert len(all_h) == 1
    assert np.array_equal(np.asarray(lh, dtype=np.float64),
                          np.asarray(legacy_lh, dtype=np.float64)), \
        "loss history drifted from the legacy loop"
    assert np.array_equal(np.asarray(best_latent), np.asarray(legacy_latent)), \
        "final latent drifted from the legacy loop"


# ---------------------------------------------------------------------------
# Synthetic costs (mirrors tests/test_msl_multistart.py)
# ---------------------------------------------------------------------------
def _bimodal_cost(x):
    shallow = 0.30 * jnp.exp(-((x - 2.0) ** 2) / (2.0 * 0.6 ** 2))
    deep = 0.90 * jnp.exp(-((x - 8.0) ** 2) / (2.0 * 0.6 ** 2))
    return 1.0 - shallow - deep


def _bimodal_cost_nan_below(x):
    return jnp.where(x < 0.5, jnp.nan, _bimodal_cost(x))


def _sharp_well(x):
    return 1.0 - jnp.exp(-((x - 8.0) ** 2) / (2.0 * 0.3 ** 2))


def _scalar(v):
    return jnp.asarray(v, dtype=jnp.float32)


# ---------------------------------------------------------------------------
# 2. The six MSL-G2 behaviors, at the library level
# ---------------------------------------------------------------------------
# Behavior 1: multi-start best-of returns the deeper (global) basin.
def test_behavior1_multistart_finds_global_min_from_both_basins():
    best_latent, lh, all_h, best_start = _adam_multistart(
        _bimodal_cost,
        [_scalar(1.5), _scalar(5.0), _scalar(8.5)],
        n_iters=120, lr=0.2, best_iterate=True,
    )
    assert abs(float(best_latent) - 8.0) < 0.15, \
        f"best_latent={float(best_latent)} should be the global min near 8.0"
    # best_start's own trajectory holds the deep-well floor.
    assert min(all_h[best_start]) < 0.2
    assert len(all_h) == 3


# Behavior 2: best-of selection is independent of seed order.
def test_behavior2_selection_independent_of_seed_order():
    _, _, _, best_start_a = _adam_multistart(
        _bimodal_cost, [_scalar(2.0), _scalar(8.0)],
        n_iters=100, lr=0.2, best_iterate=True,
    )
    best_latent_b, _, _, best_start_b = _adam_multistart(
        _bimodal_cost, [_scalar(8.0), _scalar(2.0)],
        n_iters=100, lr=0.2, best_iterate=True,
    )
    assert best_start_a == 1   # deep basin listed second wins
    assert best_start_b == 0   # deep basin listed first wins
    assert abs(float(best_latent_b) - 8.0) < 0.15


# Behavior 3: step-clamp bounds the per-step DoF (latent) move.
def test_behavior3_step_clamp_bounds_per_step_move():
    # Constant-gradient (linear) cost => Adam pushes a large, steady step.
    w = jnp.asarray([3.0, -4.0], dtype=jnp.float32)  # ||grad|| large

    def lin_cost(z):
        return jnp.sum(w * z)

    init = jnp.zeros((2,), dtype=jnp.float32)
    clamp = 0.05

    # n_iters=1 isolates a single clamped step: ||Δlatent|| <= step_clamp.
    bl1, _, _, _ = _adam_multistart(lin_cost, [init], n_iters=1, lr=5.0,
                                    step_clamp=clamp)
    move1 = float(jnp.linalg.norm(bl1 - init))
    assert move1 <= clamp + 1e-6, f"single clamped step {move1} exceeded {clamp}"

    # Unclamped: the same huge lr moves far more than the clamp would allow.
    bu1, _, _, _ = _adam_multistart(lin_cost, [init], n_iters=1, lr=5.0)
    move_unclamped = float(jnp.linalg.norm(bu1 - init))
    assert move_unclamped > clamp, "control: unclamped step should exceed clamp"

    # Cumulative bound over K clamped steps: ||final-init|| <= K*step_clamp.
    K = 6
    blK, _, _, _ = _adam_multistart(lin_cost, [init], n_iters=K, lr=5.0,
                                    step_clamp=clamp)
    assert float(jnp.linalg.norm(blK - init)) <= K * clamp + 1e-6


# Behavior 4: best-iterate returns the best-visited, not the overshot final.
def test_behavior4_best_iterate_not_overshot_final():
    init = [_scalar(7.0)]
    # High lr builds momentum and overshoots the sharp well at x=8.
    best_lat, best_lh, _, _ = _adam_multistart(
        _sharp_well, init, n_iters=40, lr=0.8, best_iterate=True,
    )
    final_lat, _, _, _ = _adam_multistart(
        _sharp_well, init, n_iters=40, lr=0.8, best_iterate=False,
    )
    c_best = float(_sharp_well(best_lat))
    c_final = float(_sharp_well(final_lat))
    # best-iterate is no worse than, and here strictly better than, final.
    assert c_best <= c_final + 1e-9
    assert c_best < c_final, "expected an overshoot (best strictly < final)"
    assert abs(float(best_lat) - 8.0) < 0.1, "best-iterate should sit at the well floor"
    # best-iterate's returned latent achieves the trajectory-min recorded loss.
    assert abs(c_best - min(best_lh)) < 1e-5


# Behavior 5: a NaN start listed first must not lock in as best.
def test_behavior5_nan_start_first_is_rejected():
    with pytest.warns(UserWarning, match="non-finite"):
        best_lat, best_lh, all_h, best_start = _adam_multistart(
            _bimodal_cost_nan_below,
            [_scalar(-1.0), _scalar(8.0)],   # NaN region first, global basin second
            n_iters=60, lr=0.2, best_iterate=True,
        )
    assert best_start == 1
    assert all_h[0] == [], "the NaN start recorded no finite loss"
    assert math.isfinite(min(best_lh)), "best loss leaked a NaN"
    assert abs(float(best_lat) - 8.0) < 0.15


# Behavior 6: all-non-finite multi-start fails loudly (fail-closed).
def test_behavior6_all_nan_multistart_raises():
    with pytest.warns(UserWarning, match="non-finite"):
        with pytest.raises(RuntimeError, match="non-finite loss at every iterate"):
            _adam_multistart(
                _bimodal_cost_nan_below,
                [_scalar(-1.0), _scalar(-2.0)],
                n_iters=20, lr=0.2,
            )


# ---------------------------------------------------------------------------
# Fail-soft (single start) + input validation
# ---------------------------------------------------------------------------
def test_single_start_nan_is_fail_soft_not_raise():
    """A SINGLE non-finite start preserves the legacy warn-and-return-last-good
    contract (does NOT raise — that is a multi-start-only guard)."""
    init = _scalar(-1.0)
    with pytest.warns(UserWarning, match="non-finite"):
        best_lat, best_lh, all_h, best_start = _adam_multistart(
            _bimodal_cost_nan_below, [init], n_iters=10, lr=0.2,
        )
    assert best_lh == []
    assert best_start == 0
    assert float(best_lat) == float(init)  # last-good == the (finite) init


def test_step_clamp_nonpositive_raises():
    with pytest.raises(ValueError, match="step_clamp must be > 0"):
        _adam_multistart(_bowl_cost, [jnp.zeros((3,), jnp.float32)],
                         n_iters=1, lr=0.1, step_clamp=0.0)


def test_helper_is_pure_no_fdtd():
    """`_adam_multistart` is importable and runs with no rfx forward."""
    bl, lh, all_h, bs = _adam_multistart(
        lambda z: jnp.sum((z - 3.0) ** 2),
        [jnp.asarray(3.0, dtype=jnp.float32)],
        n_iters=1, lr=0.1,
    )
    assert abs(float(bl) - 3.0) < 1e-3   # already at the min
    assert bs == 0 and len(all_h) == 1


# ---------------------------------------------------------------------------
# End-to-end through optimize() — FDTD (gpu-marked like the other optimize tests)
# ---------------------------------------------------------------------------
def _tiny_sim():
    from rfx.api import Simulation
    sim = Simulation(freq_max=5e9, domain=(0.015, 0.015, 0.015), boundary="pec")
    sim.add_port((0.005, 0.0075, 0.0075), "ez")
    sim.add_probe((0.011, 0.0075, 0.0075), "ez")
    return sim


def _tiny_region():
    return DesignRegion(
        corner_lo=(0.006, 0.006, 0.006),
        corner_hi=(0.010, 0.010, 0.009),
        eps_range=(1.0, 4.4),
    )


def _obj(r):
    return -jnp.sum(r.time_series ** 2)


@pytest.mark.gpu
def test_optimize_default_bit_identity_end_to_end():
    """optimize() with the new params at their defaults is byte-identical to
    optimize() with the params omitted (and seed is irrelevant at n_starts=1)."""
    region = _tiny_region()
    res_a = optimize(_tiny_sim(), region, _obj, n_iters=5, lr=0.2, verbose=False)
    res_b = optimize(_tiny_sim(), region, _obj, n_iters=5, lr=0.2, verbose=False,
                     n_starts=1, best_iterate=False, step_clamp=None, seed=12345)
    assert np.array_equal(np.asarray(res_a.loss_history, dtype=np.float64),
                          np.asarray(res_b.loss_history, dtype=np.float64))
    assert np.array_equal(np.asarray(res_a.latent), np.asarray(res_b.latent))
    assert np.array_equal(np.asarray(res_a.eps_design), np.asarray(res_b.eps_design))


@pytest.mark.gpu
def test_optimize_multistart_is_deterministic_under_fixed_seed():
    """Same seed => identical multi-start result (explicit, reproducible RNG)."""
    region = _tiny_region()
    r1 = optimize(_tiny_sim(), region, _obj, n_iters=3, lr=0.2, verbose=False,
                  n_starts=2, seed=7)
    r2 = optimize(_tiny_sim(), region, _obj, n_iters=3, lr=0.2, verbose=False,
                  n_starts=2, seed=7)
    assert np.array_equal(np.asarray(r1.loss_history, dtype=np.float64),
                          np.asarray(r2.loss_history, dtype=np.float64))
    assert np.array_equal(np.asarray(r1.latent), np.asarray(r2.latent))


@pytest.mark.gpu
def test_optimize_multistart_and_knobs_plumb_through():
    """n_starts>1 / best_iterate / step_clamp all run and return a valid result."""
    region = _tiny_region()
    res = optimize(_tiny_sim(), region, _obj, n_iters=3, lr=0.2, verbose=False,
                   n_starts=2, best_iterate=True, step_clamp=1e-2, seed=1)
    assert isinstance(res, OptimizeResult)
    assert len(res.loss_history) >= 1
    assert np.all(np.isfinite(np.asarray(res.eps_design)))
    assert res.eps_design.shape == res.latent.shape


def test_optimize_rejects_bad_n_starts():
    # Raised before any FDTD work, so this stays in the fast (non-gpu) suite.
    with pytest.raises(ValueError, match="n_starts must be >= 1"):
        optimize(_tiny_sim(), _tiny_region(), _obj, n_iters=1, verbose=False,
                 n_starts=0)
