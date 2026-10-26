"""Nested checkpointing in ``rfx.stepping.segmented_scan`` and its planner.

The toy is a 2-D leap-frog grid (Ez, Hx, Hy on 40 x 40 float32 cells, an
initial Ez bump and a point source) with a per-cell update coefficient as
the differentiable parameter, so the reverse pass needs a per-step record.
Lengths are outermost first. The one-level reference is BASE 1d5712706.
"""

import json
import sys

import jax
import jax.extend.core as jex
import jax.numpy as jnp
import numpy as np
import pytest

from rfx.stepping import plan_checkpoint_lengths, segmented_scan
from rfx.stepping.remat import checkpoint_lengths, checkpoint_memory


# ---------------------------------------------------------------------------
# Reference: ``effective_chunk`` and ``segmented_scan`` copied verbatim from
# rfx/stepping/remat.py at commit 1d5712706 (before nested checkpointing).
def _main_effective_chunk(n, chunk):
    """The segment length a remat over ``n`` steps uses, or None (no remat).

    A chunk above n/2 leaves one full segment and a tail; XLA then keeps the
    whole per-step tape (measured: chunk 61 of 120 steps used the memory of
    no remat). Such a chunk is clamped to n // 2, so there are always at
    least two full segments. Only memory and compute depend on it, never a
    result.
    """
    if chunk is None or not (0 < chunk < n):
        return None
    return min(int(chunk), max(1, n // 2))


def _main_segmented_scan(step_fn, carry, xs, chunk, scan=jax.lax.scan):
    """Scan a pytree record in checkpointed chunks and a real partial tail.

    ``scan`` controls the outer scan, allowing callers to retain recording
    hooks. Inner scans always use ``jax.lax.scan``. Inactive segmentation
    delegates directly to ``scan`` without adding a checkpoint wrapper.
    """
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


# The tail of the outermost level is split again by the inner levels.
def _tail_unit_depth(chunk, n):
    """Checkpoint nesting depth inside the outermost tail unit only: the
    top-level scan of length 1 that runs the r leftover steps."""
    jaxpr = jax.make_jaxpr(_loss(segmented_scan, chunk, n))(COEF).jaxpr
    tails = [eqn for eqn in jaxpr.eqns
             if eqn.primitive.name == "scan" and eqn.params["length"] == 1]
    assert len(tails) == 1, [e.params.get("length") for e in jaxpr.eqns
                             if e.primitive.name == "scan"]
    return _remat_depth(tails[0].params["jaxpr"].jaxpr)


# Expected depth = 1 (the tail unit's own checkpoint) + the inner levels the
# clamp leaves active on the tail's r steps, worked out by hand:
@pytest.mark.parametrize("chunk,n,expected", [
    ((24, 6, 2), 118, 3),  # r = 22: 6 active on 22, 2 active on 6
    ((24, 6, 2), 100, 2),  # r = 4: 6 skipped on 4, 2 active on 4
    ((36, 6), 287, 2),     # r = 35: 6 active on 35
    ((24, 6, 2), 97, 1),   # r = 1: no inner level fits one step
])
def test_outer_tail_is_split_by_the_inner_levels(chunk, n, expected):
    assert _tail_unit_depth(chunk, n) == expected


# Memory: compiled temp bytes, (L1, L2) against (L1,) with the same outer
# length run as one level. On CPU this is a report of XLA's static analysis;
# the device gate for the memory claim is the GPU suite.
def test_two_levels_use_less_compiled_memory_than_their_outer_level(
        record_property):
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
        except Exception:  # backend without memory analysis
            return None
        value = getattr(analysis, "temp_size_in_bytes", None)
        return None if value is None else int(value)

    outer = (two.lengths[0],)
    nested, flat = temp_bytes(two.lengths), temp_bytes(outer)
    if not nested and not flat:
        pytest.skip(f"{jax.default_backend()} memory_analysis() gives no temp "
                    "bytes; the device gate is the GPU suite")
    report = dict(
        nested=nested, nested_lengths=two.lengths, outer_only=flat,
        outer_lengths=outer, planner_one_level=temp_bytes(one.lengths),
        planner_one_level_lengths=one.lengths, plain=temp_bytes(None),
        model_one_level=one.peak_bytes, model_two_levels=two.peak_bytes)
    record_property("temp_bytes", report)
    sys.stderr.write(f"compiled temp bytes: {report}\n")
    assert nested is not None and flat is not None, report
    assert nested < flat, report


# Planner.
BYTES = dict(boundary_bytes=1.45e6, step_bytes=3.85e6, static_bytes=2.0e8)


def _assert_clamp_leaves_lengths(n, lengths):
    # Independent restatement of the clamp: every level keeps at least two
    # full units of its enclosing span and is shorter than that span.
    enclosing = n
    for length in lengths:
        assert 0 < length < enclosing and length <= enclosing // 2, (n, lengths)
        enclosing = length


# Boundary carries far dearer than a step: the least peak sits at the clamp's
# edge (two full units), where a length above n // 2 would hold fewer carries.
BYTES_EDGE = dict(boundary_bytes=1.0e8, step_bytes=1.0e6, static_bytes=0.0)


@pytest.mark.parametrize("byte_counts", (BYTES, BYTES_EDGE), ids=("fit", "edge"))
@pytest.mark.parametrize("n", (4, 17, 97, 288, 1000, 19092))
def test_planner_lengths_pass_validation_and_the_clamp(n, byte_counts):
    for budget in (1e8, 3e8, 1e9, 1e10, 1e12):
        for max_levels in (1, 2, 3):
            plan = plan_checkpoint_lengths(n, budget_bytes=budget,
                                           max_levels=max_levels, **byte_counts)
            assert plan.lengths is not None
            assert checkpoint_lengths(plan.lengths) == plan.lengths
            _assert_clamp_leaves_lengths(n, plan.lengths)
            assert plan.levels == len(plan.lengths) <= max_levels
            assert plan.forward_passes == 1 + plan.levels
            assert checkpoint_memory(n, plan.lengths, **byte_counts).lengths == plan.lengths


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
