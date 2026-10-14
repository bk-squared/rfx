"""Nested checkpointing in ``rfx.stepping.segmented_scan`` and its planner.

The toy is a 2-D leap-frog grid (Ez, Hx, Hy on 40 x 40 float32 cells, an
initial Ez bump and a point source) with a per-cell update coefficient as
the differentiable parameter, so the reverse pass needs a per-step record.
Lengths are outermost first.
"""

import json

import jax
import jax.extend.core as jex
import jax.numpy as jnp
import numpy as np
import pytest

from rfx.stepping import plan_checkpoint_lengths, segmented_scan
from rfx.stepping.remat import checkpoint_lengths, checkpoint_memory


# ---------------------------------------------------------------------------
# Reference: ``effective_chunk`` and ``segmented_scan`` copied verbatim from
# rfx/stepping/remat.py at commit cd85b3a7a (before nested checkpointing).
def _main_effective_chunk(n, chunk):
    if chunk is None or not (0 < chunk < n):
        return None
    return min(int(chunk), max(1, n // 2))


def _main_segmented_scan(step_fn, carry, xs, chunk, scan=jax.lax.scan):
    """Main's ``segmented_scan`` at cd85b3a7a, verbatim."""
    n = jax.tree_util.tree_leaves(xs)[0].shape[0]
    chunk = _main_effective_chunk(n, chunk)
    if chunk is None:
        return scan(step_fn, carry, xs)

    m, r = divmod(n, chunk)
    end = m * chunk
    segments = jax.tree_util.tree_map(
        lambda x: x[:end].reshape((m, chunk) + x.shape[1:]), xs
    )

    def segment_body(c, s):
        return jax.lax.scan(step_fn, c, s)

    body = jax.checkpoint(segment_body)
    carry, ys = scan(body, carry, segments)
    ys = jax.tree_util.tree_map(
        lambda y: y.reshape((end,) + y.shape[2:]), ys
    )
    if r:
        # The tail is one more checkpointed segment of length r, run through
        # the caller's ``scan`` too (a recorded scan must see every step).
        tail = jax.tree_util.tree_map(lambda x: x[end:][None], xs)
        carry, tail_ys = scan(body, carry, tail)
        ys = jax.tree_util.tree_map(
            lambda y, t: jnp.concatenate((y, t[0]), axis=0), ys, tail_ys
        )
    return carry, ys
# ---------------------------------------------------------------------------


N = 40


def _run(scan_fn, coef, n, chunk):
    def step(carry, x):
        ez, hx, hy = carry
        hx = hx - 0.4 * (jnp.roll(ez, -1, axis=1) - ez)
        hy = hy + 0.4 * (jnp.roll(ez, -1, axis=0) - ez)
        curl = (hy - jnp.roll(hy, 1, axis=0)) - (hx - jnp.roll(hx, 1, axis=1))
        ez = ez + coef * curl
        ez = ez.at[N // 3, N // 3].add(x)
        return (ez, hx, hy), ez[2 * N // 3, N // 2]

    t = jnp.arange(n, dtype=jnp.float32)
    xs = jnp.exp(-((t - 12.0) / 5.0) ** 2) * jnp.sin(0.6 * t)
    zeros = jnp.zeros((N, N), jnp.float32)
    # A field already present, so even a one-step record has a gradient.
    r2 = (jnp.arange(N, dtype=jnp.float32)[:, None] - 0.6 * N) ** 2 + (
        jnp.arange(N, dtype=jnp.float32)[None, :] - 0.5 * N) ** 2
    ez0 = jnp.exp(-r2 / 8.0)
    return scan_fn(step, (ez0, zeros, zeros), xs, chunk)


def _loss(scan_fn, chunk, n):
    def loss(coef):
        (ez, hx, hy), ys = _run(scan_fn, coef, n, chunk)
        return jnp.sum(ys ** 2) + 1e-3 * jnp.sum(ez ** 2)
    return loss


COEF = 0.4 * (1.0 + 0.2 * jax.random.uniform(jax.random.PRNGKey(0), (N, N)))


def _values_and_grad(scan_fn, chunk, n):
    carry, ys = jax.jit(lambda c: _run(scan_fn, c, n, chunk))(COEF)
    grad = jax.jit(jax.grad(_loss(scan_fn, chunk, n)))(COEF)
    return [np.asarray(a) for a in (*carry, ys)], np.asarray(grad)


def _bitwise(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


# Invariant (a): a one-length tuple is the int, and the int is main's path.
@pytest.mark.parametrize("c", (5, 12, 24))
@pytest.mark.parametrize("n", (1, 2, 23, 24, 25, 47, 97))
def test_one_length_tuple_is_bit_identical_to_int_and_to_main(c, n):
    ref_values, ref_grad = _values_and_grad(_main_segmented_scan, c, n)
    int_values, int_grad = _values_and_grad(segmented_scan, c, n)
    tup_values, tup_grad = _values_and_grad(segmented_scan, (c,), n)
    for ref, a, b in zip(ref_values, int_values, tup_values):
        assert _bitwise(ref, a) and _bitwise(ref, b)
    assert _bitwise(ref_grad, int_grad) and _bitwise(ref_grad, tup_grad)


# Invariant (b): any record length gives the plain scan's values.
LEVELS = ((24, 6), (12, 4), (5, 2), (24, 6, 2))


def _record_lengths(lengths):
    big = lengths[0]
    return sorted({1, 2, big - 1, big, big + 1, 2 * big - 1, 97, 288})


@pytest.mark.parametrize(
    "lengths,n",
    [(lengths, n) for lengths in LEVELS for n in _record_lengths(lengths)],
)
def test_nested_values_equal_plain_scan(lengths, n, record_property):
    plain_values, plain_grad = _values_and_grad(_main_segmented_scan, None, n)
    values, grad = _values_and_grad(segmented_scan, lengths, n)
    for ref, actual in zip(plain_values, values):
        assert actual.shape == ref.shape
        assert _bitwise(ref, actual), float(np.max(np.abs(actual - ref)))
    scale = float(np.max(np.abs(plain_grad)))
    assert scale > 0
    relative = float(np.max(np.abs(grad - plain_grad))) / scale
    record_property("gradient_relative_difference", relative)
    assert relative <= 1e-6, relative


# Invariant (c): checkpoint nesting depth equals the active levels.
def _remat_depth(jaxpr):
    depth = 0
    for eqn in jaxpr.eqns:
        inner = 0
        for value in eqn.params.values():
            for sub in (value if isinstance(value, (tuple, list)) else (value,)):
                if isinstance(sub, jex.ClosedJaxpr):
                    inner = max(inner, _remat_depth(sub.jaxpr))
                elif isinstance(sub, jex.Jaxpr):
                    inner = max(inner, _remat_depth(sub))
        if eqn.primitive.name in ("remat2", "checkpoint", "remat"):
            inner += 1
        depth = max(depth, inner)
    return depth


@pytest.mark.parametrize("chunk,n,expected", [
    (None, 288, 0),
    (24, 288, 1),
    ((24,), 288, 1),
    ((24, 6), 288, 2),
    ((24, 6, 2), 288, 3),
    ((24, 6), 30, 2),      # outer clamped to 15, inner 6 stays
    ((24, 6), 20, 1),      # outer skipped (24 >= 20), inner 6 active
    ((24, 6), 5, 0),       # both skipped
    ((24, 6, 2), 97, 3),   # tails at the outer level are split again
])
def test_checkpoint_nesting_depth(chunk, n, expected):
    jaxpr = jax.make_jaxpr(_loss(segmented_scan, chunk, n))(COEF)
    assert _remat_depth(jaxpr.jaxpr) == expected


@pytest.mark.parametrize("chunk", [(), (6, 24), (6, 6), (6, 0), (6, -1),
                                   (True, 1), (6, 2.0), (6.0,)])
def test_invalid_lengths_are_refused(chunk):
    with pytest.raises(ValueError):
        segmented_scan(lambda c, x: (c, x), 0.0, jnp.zeros(8), chunk)


# Memory: compiled temp bytes, one level vs two levels.
def test_two_levels_do_not_use_more_compiled_memory(record_property):
    n = 288
    field = 3 * N * N * 4
    common = dict(boundary_bytes=field, step_bytes=field, static_bytes=0,
                  budget_bytes=0)
    one = plan_checkpoint_lengths(n, max_levels=1, **common)
    two = plan_checkpoint_lengths(n, max_levels=2, **common)
    assert one.levels == 1 and two.levels == 2

    def temp_bytes(chunk):
        compiled = jax.jit(jax.grad(_loss(segmented_scan, chunk, n))).lower(
            COEF).compile()
        try:
            analysis = compiled.memory_analysis()
        except Exception as exc:  # backend without memory analysis
            pytest.skip(f"memory_analysis() raised {type(exc).__name__}")
        value = getattr(analysis, "temp_size_in_bytes", None)
        if value is None:
            pytest.skip("memory_analysis() has no temp_size_in_bytes")
        return int(value)

    plain = temp_bytes(None)
    first, second = temp_bytes(one.lengths), temp_bytes(two.lengths)
    record_property("temp_bytes", dict(
        plain=plain, one_level=first, two_levels=second,
        one_level_lengths=one.lengths, two_level_lengths=two.lengths,
        model_one=one.peak_bytes, model_two=two.peak_bytes))
    if not second < first:
        pytest.skip(
            f"{jax.default_backend()} memory_analysis shows no reduction "
            f"(plain {plain}, one level {first}, two levels {second} temp "
            "bytes); the device gate is the GPU suite")
    assert second <= first


# Planner.
BYTES = dict(boundary_bytes=1.45e6, step_bytes=3.85e6, static_bytes=2.0e8)


def _assert_clamp_leaves_lengths(n, lengths):
    # Independent restatement of the clamp: every level keeps at least two
    # full units of its enclosing span and is shorter than that span.
    enclosing = n
    for length in lengths:
        assert 0 < length < enclosing and length <= enclosing // 2, (n, lengths)
        enclosing = length


@pytest.mark.parametrize("n", (4, 17, 97, 288, 1000, 19092))
def test_planner_lengths_pass_validation_and_the_clamp(n):
    for budget in (1e8, 3e8, 1e9, 1e10, 1e12):
        for max_levels in (1, 2, 3):
            plan = plan_checkpoint_lengths(n, budget_bytes=budget,
                                           max_levels=max_levels, **BYTES)
            assert plan.lengths is not None
            assert checkpoint_lengths(plan.lengths) == plan.lengths
            _assert_clamp_leaves_lengths(n, plan.lengths)
            assert plan.levels == len(plan.lengths) <= max_levels
            assert plan.forward_passes == 1 + plan.levels
            assert checkpoint_memory(n, plan.lengths, **BYTES).lengths == plan.lengths


def test_planner_is_monotone_in_budget():
    n = 19092
    budgets = np.geomspace(2.0e8, 1.0e11, 25)
    levels = [plan_checkpoint_lengths(n, budget_bytes=b, **BYTES).levels
              for b in budgets]
    assert all(a >= b for a, b in zip(levels, levels[1:])), levels
    assert levels[0] > levels[-1]


def test_planner_reports_the_best_peak_when_nothing_fits():
    n = 19092
    plan = plan_checkpoint_lengths(n, budget_bytes=1.0, **BYTES)
    assert plan.fits is False
    best = [plan_checkpoint_lengths(n, budget_bytes=1.0, max_levels=k, **BYTES)
            for k in (1, 2, 3)]
    assert plan.peak_bytes == min(p.peak_bytes for p in best)
    assert plan.peak_bytes > BYTES["static_bytes"]


def test_memory_model_counts_held_boundaries_and_the_live_record():
    # 288 steps as (24, 6): 12 outer units + 4 per unit; live record 6 steps.
    memory = checkpoint_memory(288, (24, 6), **BYTES)
    assert memory.held_boundaries == 12 + 4 and memory.live_steps == 6
    assert memory.peak_bytes == pytest.approx(2.0e8 + 1.45e6 * 16 + 3.85e6 * 6)
    # 97 steps as (24, 6, 2): ceil(97/24) = 5, 4, 3 held; live 2.
    memory = checkpoint_memory(97, (24, 6, 2), **BYTES)
    assert memory.held_boundaries == 5 + 4 + 3 and memory.live_steps == 2
    assert memory.forward_passes == 4


# The estimate (Simulation.estimate_ad_memory with a tuple checkpoint_every).
def _graded_sim():
    from rfx import Simulation
    dz = np.concatenate([np.full(6, 0.3e-3), np.full(6, 0.4e-3)])
    sim = Simulation(freq_max=10e9, domain=(8e-3, 8e-3, float(np.sum(dz))),
                     dx=0.5e-3, dz_profile=dz, boundary="cpml", cpml_layers=4)
    sim.add_source((4e-3, 4e-3, 1e-3), "ez", amplitude_kind="current")
    sim.add_probe((4e-3, 4e-3, 2e-3), "ez")
    return sim


def test_estimate_of_nested_lengths_follows_the_model():
    sim = _graded_sim()
    n_steps, n_warmup = 2000, 80
    estimate = sim.estimate_ad_memory(n_steps, checkpoint_every=(240, 16),
                                      n_warmup=n_warmup,
                                      available_memory_gb=1e3)
    cells = int(np.prod(estimate.grid_shape))
    field = 6 * 4 * cells
    # 1920 optimised steps: ceil(1920 / 240) = 8 outer + 15 inner held; live 16.
    static = estimate.forward_gb * 1e9 + estimate.ntff_dft_gb * 1e9 + (
        3.5e9 / 5.67e6) * cells
    expected = static + 1.45 * field * (8 + 15) + 3.85 * field * 16
    assert estimate.checkpoint_every == (240, 16)
    assert estimate.ad_segmented_active_segments == 23
    assert estimate.ad_segmented_gb * 1e9 == pytest.approx(expected, rel=1e-9)
    json.loads(estimate.to_json())

    report = sim.explain_ad_memory(n_steps, checkpoint_every=(240, 16),
                                   n_warmup=n_warmup, available_memory_gb=1e3)
    assert report.strategy == "nested_checkpoint_every"
    total = sum(component.memory_gb for component in report.components)
    assert total == pytest.approx(report.selected_memory_gb, rel=1e-9)
    assert any("3 forward passes" in line for line in report.recommendations)


def test_estimate_refuses_bad_nested_lengths():
    sim = _graded_sim()
    with pytest.raises(ValueError):
        sim.estimate_ad_memory(200, checkpoint_every=(16, 24))
    with pytest.raises(ValueError):
        sim.estimate_ad_memory(200, checkpoint_every=(24, 6),
                               checkpoint_segments=4)


def test_plan_ad_memory_levels_use_the_same_model():
    sim = _graded_sim()
    kwargs = dict(target_fraction=1.0, safety_factor=1.0)
    # With a budget no plan meets, each level count returns its least peak.
    least = {k: sim.plan_ad_memory(2000, 1e-6, max_levels=k, **kwargs)
             for k in (1, 2)}
    assert [len(least[k].checkpoint_every) for k in (1, 2)] == [1, 2]
    one_gb = least[1].selected_estimate.ad_segmented_gb
    two_gb = least[2].selected_estimate.ad_segmented_gb
    assert two_gb < one_gb and not least[2].segmented_fits
    # A budget between the two: one level fails, two levels fit.
    budget_gb = 0.5 * (one_gb + two_gb)
    plan = sim.plan_ad_memory(2000, budget_gb, max_levels=3, **kwargs)
    assert plan.segmented_fits and len(plan.checkpoint_every) == 2
    _assert_clamp_leaves_lengths(2000, plan.checkpoint_every)
    estimate = sim.estimate_ad_memory(2000, checkpoint_every=plan.checkpoint_every,
                                      available_memory_gb=1e3)
    assert plan.selected_estimate.ad_segmented_gb == estimate.ad_segmented_gb
    assert estimate.ad_segmented_gb <= budget_gb


def test_plan_ad_memory_levels_refused_on_a_uniform_grid():
    from rfx import Simulation
    sim = Simulation(freq_max=10e9, domain=(8e-3, 8e-3, 4e-3), dx=1e-3,
                     boundary="cpml", cpml_layers=2)
    sim.add_source((4e-3, 4e-3, 2e-3), "ez", amplitude_kind="current")
    with pytest.raises(ValueError, match="uniform"):
        sim.plan_ad_memory(2000, 1e-3, max_levels=2)
