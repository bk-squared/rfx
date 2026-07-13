"""Unit tests for the best-of multi-start Adam helper in the MSL
open-stub notch-tuning example (issue #171 close-out).

The MSL ``|S21(f_target)|²`` cost over L_stub ∈ [4, 12] mm is physically
MULTIMODAL (in-band λ/4 notch ~7.0 mm = global min vs the below-band
longer-stub valley ~9.5 mm).  A single-start Adam can settle in the
secondary valley — the latent defect the #171 falsifier identified.
``_multistart_adam`` fixes it by running Adam from band-spanning seeds
and keeping the best-of (lowest final cost) trajectory, with a per-step
clamp that bounds the physical-length move so a high lr cannot overshoot
a valley wall.

These tests exercise that helper against a SYNTHETIC bimodal cost — no
FDTD, no rfx forward — so the optimizer logic is verified in isolation.
The example module is import-safe (``main()`` is ``__main__``-guarded),
so we load it via ``importlib.util`` and reach into the module-level
helper.
"""

from __future__ import annotations

import importlib.util
import math
import pathlib

import pytest
import jax
import jax.numpy as jnp

# NOTE: do NOT call jax.config.update("jax_enable_x64", True) at module
# level — it is a PROCESS-GLOBAL flag, and flipping it at import/collection
# time contaminates every other test in the same pytest-split shard
# (their complex64 scan carries then receive complex128 DFT accumulators →
# "carry input and output must have equal types" TypeError). These tests
# exercise the pure float32 _multistart_adam helper, which is exactly how
# the example runs in production; tolerances below are float32-safe.

EXAMPLE = (
    pathlib.Path(__file__).resolve().parent.parent
    / "validation" / "tap_paper" / "msl_stub_notch_tuning.py"
)


def _load_example():
    """Execute the example module top level (does NOT run ``main``)."""
    spec = importlib.util.spec_from_file_location(
        "_msl_stub_notch_tuning", EXAMPLE
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _bimodal_cost(x):
    """Synthetic bimodal cost on a scalar (identity-latent) variable.

    Two smooth wells: a SHALLOW local min near x = 2.0 and a DEEP
    GLOBAL min near x = 8.0.  Built as a product of two Gaussians
    subtracted from a constant so it is smooth + differentiable
    everywhere and ``jax.value_and_grad`` flows cleanly.

      cost(x) = 1 - 0.30*exp(-(x-2)²/(2*0.6²))   # shallow well @ 2.0
                  - 0.90*exp(-(x-8)²/(2*0.6²))   # deep   well @ 8.0

    Global minimum is at x = 8.0 (cost ≈ 0.10); the local minimum at
    x = 2.0 bottoms at cost ≈ 0.70.  Well width ~0.6 in x.
    """
    shallow = 0.30 * jnp.exp(-((x - 2.0) ** 2) / (2.0 * 0.6 ** 2))
    deep = 0.90 * jnp.exp(-((x - 8.0) ** 2) / (2.0 * 0.6 ** 2))
    return 1.0 - shallow - deep


def test_multistart_finds_global_min_from_both_basins():
    """Seeds in BOTH basins → best-of returns the deeper (global) min.

    A single start from the shallow basin would stay at x≈2.0.  With
    one seed in each basin, the best-of (lowest final cost) trajectory
    must be the one that settled in the deep global well near x=8.0.
    """
    module = _load_example()
    fn = module._multistart_adam

    # Identity latent map: the optimizer variable IS the physical length.
    latent_to_L = lambda lat: lat  # noqa: E731

    # One seed in the shallow basin (near 2.0), one in the global basin
    # (near 8.0), plus a neutral midpoint.
    latent_inits = [1.5, 5.0, 8.5]

    best_L, best_cost, best_history, all_histories = fn(
        cost_fn=_bimodal_cost,
        latent_inits=latent_inits,
        n_iters=120,
        lr=0.2,
        max_dL_per_step=0.5,   # generous — does not block reaching 8.0
        latent_to_L=latent_to_L,
    )

    # Best-of must land in the GLOBAL well near x = 8.0, not the
    # shallow x = 2.0 local min.
    assert abs(best_L - 8.0) < 0.15, (
        f"best_L={best_L} should be the global min near 8.0, not the "
        f"shallow trap near 2.0"
    )
    # And its cost must be the deep-well floor, well below the shallow
    # local min (~0.70).
    assert best_cost < 0.2, f"best_cost={best_cost} not at the deep-well floor"

    # best_cost must equal the lowest BEST-ITERATE cost across all starts
    # (not the lowest final — selection is on the best visited point).
    bests = [h["cost_best"] for h in all_histories]
    assert best_cost == min(bests)
    assert len(all_histories) == len(latent_inits)


def test_multistart_picks_deeper_even_when_shallow_seed_listed_first():
    """Best-of is by final cost, independent of seed order."""
    module = _load_example()
    fn = module._multistart_adam
    latent_to_L = lambda lat: lat  # noqa: E731

    # Only two seeds, shallow basin listed FIRST.
    best_L, best_cost, best_history, all_histories = fn(
        cost_fn=_bimodal_cost,
        latent_inits=[2.0, 8.0],
        n_iters=100,
        lr=0.2,
        max_dL_per_step=0.5,
        latent_to_L=latent_to_L,
    )
    assert abs(best_L - 8.0) < 0.15
    assert best_history["start"] == 1  # the deep-basin seed won


def test_step_clamp_bounds_physical_move_per_iter():
    """No recorded per-step |ΔL| exceeds ``max_dL_per_step``.

    With a deliberately huge lr the unclamped Adam step would jump far
    in one iteration.  The clamp must hold every consecutive L move
    (and the init→first-iter move) to within max_dL_per_step.
    """
    module = _load_example()
    fn = module._multistart_adam
    latent_to_L = lambda lat: lat  # noqa: E731

    max_dL = 0.1
    seed = 5.0  # far from both wells → large gradient/step pressure
    best_L, best_cost, best_history, all_histories = fn(
        cost_fn=_bimodal_cost,
        latent_inits=[seed],
        n_iters=30,
        lr=5.0,                 # huge — unclamped step would overshoot
        max_dL_per_step=max_dL,
        latent_to_L=latent_to_L,
    )

    h = all_histories[0]
    # The recorded L values are at the START of each iter (pre-update).
    # h["L"][0] is the seed; consecutive deltas are the applied steps.
    # h["L"] is in MILLIMETRES of latent_to_L output * 1e3 in the
    # example; here latent_to_L is identity so h["L"] = L * 1e3.
    Ls = [v / 1e3 for v in h["L"]]
    assert abs(Ls[0] - seed) < 1e-6
    tol = 1e-5  # float32-safe clamp slack (the clamp is exact; this only
    #            absorbs float32 rounding in the L round-trip)
    for a, b in zip(Ls[:-1], Ls[1:]):
        assert abs(b - a) <= max_dL + tol, (
            f"per-step |ΔL|={abs(b-a)} exceeded clamp {max_dL}"
        )
    # best_L is the BEST-seen iterate — it must be one of the visited L
    # points (the clamp bounds the moves between them, asserted above).
    assert any(abs(best_L - L) < 1e-6 for L in Ls), (
        f"best_L={best_L} is not a visited iterate"
    )


def _bimodal_cost_nan_below(x):
    """``_bimodal_cost`` but NaN for x < 0.5 — models the documented
    deep-notch NaN region (the kind PR #170 fixed on the grad path).
    A seed placed here finals to a NaN cost.
    """
    return jnp.where(x < 0.5, jnp.nan, _bimodal_cost(x))


def test_multistart_skips_nan_start_listed_first():
    """Review Finding 1 (MAJOR) regression: a NaN final cost from the
    FIRST start must NOT lock in as 'best'.

    Old code (``best_cost is None or final_loss < best_cost``) set
    best_cost to the first start's NaN, after which every finite
    challenger failed ``finite < nan`` and the helper returned a NaN
    L_opt that the dB printout masked.  The NaN-safe selection must
    instead return the finite global optimum.
    """
    module = _load_example()
    fn = module._multistart_adam
    latent_to_L = lambda lat: lat  # noqa: E731

    # NaN-producing seed listed FIRST; finite global-basin seed second.
    best_L, best_cost, best_history, all_histories = fn(
        cost_fn=_bimodal_cost_nan_below,
        latent_inits=[-1.0, 8.0],
        n_iters=60,
        lr=0.2,
        max_dL_per_step=0.5,
        latent_to_L=latent_to_L,
    )
    # The first start's final cost is NaN; it must be rejected.
    assert not math.isfinite(all_histories[0]["cost_final"])
    # Best-of must be the finite global optimum (the second start).
    assert math.isfinite(best_cost), f"best_cost={best_cost} leaked a NaN"
    assert best_history["start"] == 1
    assert abs(best_L - 8.0) < 0.15


def test_multistart_all_nan_raises():
    """If every start finals to a non-finite cost, fail loudly rather
    than indexing all_histories[None] and returning a NaN optimum."""
    module = _load_example()
    fn = module._multistart_adam
    latent_to_L = lambda lat: lat  # noqa: E731
    with pytest.raises(RuntimeError, match="non-finite cost at every iterate"):
        fn(
            cost_fn=_bimodal_cost_nan_below,
            latent_inits=[-1.0, -2.0],
            n_iters=20,
            lr=0.2,
            max_dL_per_step=0.5,
            latent_to_L=latent_to_L,
        )


def _sharp_well(x):
    """A NARROW global well at x=8.0 that Adam with momentum overshoots.

    cost(x) = 1 - exp(-(x-8)^2 / (2*sigma^2)),  sigma=0.3
    Smooth/differentiable; sharp enough that a high-lr Adam rushes in,
    hits the ~0 floor near x=8, and momentum carries it back out — so the
    FINAL iterate is worse than the BEST visited iterate.
    """
    return 1.0 - jnp.exp(-((x - 8.0) ** 2) / (2.0 * 0.3 ** 2))


def test_multistart_keeps_best_iterate_not_overshot_final():
    """GPU run 369367242483 regression (#171): on a sharp resonant null
    Adam OVERSHOOTS (hit -46 dB at L=7.0-7.1mm, momentum carried it out to
    7.43mm/-34 dB), and returning the FINAL iterate discarded the deep
    point Adam actually visited. The helper must return the BEST-seen
    iterate along the trajectory.
    """
    module = _load_example()
    fn = module._multistart_adam
    latent_to_L = lambda lat: lat  # noqa: E731

    best_L, best_cost, best_history, all_histories = fn(
        cost_fn=_sharp_well,
        latent_inits=[7.0],
        n_iters=40,
        lr=0.8,                 # high — builds momentum, overshoots the well
        max_dL_per_step=10.0,   # effectively unclamped
        latent_to_L=latent_to_L,
    )
    # Invariant (always holds): the kept optimum is the min over the whole
    # trajectory, and is no worse than the final iterate.
    h = best_history
    finite_costs = [c for c in h["cost"] if math.isfinite(c)]
    assert best_cost == min(finite_costs)
    assert h["cost_best"] <= h["cost_final"] + 1e-12

    # This setup actually overshoots: the best-seen is the well floor near
    # x=8, strictly better than the overshot final, at a different L.
    assert abs(best_L - 8.0) < 0.1, f"best_L={best_L} not at the well floor"
    assert h["cost_best"] < h["cost_final"], (
        "expected an overshoot (best < final); if this fails the test "
        "setup no longer exercises the overshoot path"
    )
    assert abs(h["L_best"] - h["L_final"]) > 1e-3


def test_helper_is_importable_and_pure():
    """``_multistart_adam`` exists at module level and needs no FDTD."""
    module = _load_example()
    assert hasattr(module, "_multistart_adam")
    assert hasattr(module, "main")  # main stays __main__-guarded
    # Trivial single-start, single-iter run with a constant cost: must
    # not touch any rfx forward and must return the seed unchanged-ish.
    best_L, best_cost, _, all_h = module._multistart_adam(
        cost_fn=lambda x: (x - 3.0) ** 2,
        latent_inits=[3.0],
        n_iters=1,
        lr=0.1,
        max_dL_per_step=0.5,
        latent_to_L=lambda lat: lat,
    )
    assert abs(best_L - 3.0) < 1e-3  # already at the min, no movement
    assert best_cost < 1e-6
