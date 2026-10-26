"""Segment-level rematerialization without extending the time record."""

import math
import numbers
from typing import NamedTuple

import jax
import jax.numpy as jnp


def effective_chunk(n, chunk):
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


def active_levels(n, lengths):
    """The lengths of the levels a nested scan over ``n`` steps runs.

    Each level is clamped by :func:`effective_chunk` on the span of the level
    above it; a level the clamp turns off is skipped and the inner levels
    still run, exactly as :func:`segmented_scan` does.
    """
    active, span = [], n
    for length in lengths:
        used = effective_chunk(span, length)
        if used is not None:
            active.append(used)
            span = used
    return tuple(active)


def checkpoint_lengths(chunk):
    """Validate a tuple of checkpoint lengths, outermost first.

    Every length is a positive int (a bool is refused) and the tuple is
    strictly decreasing. Returns the lengths as a tuple of Python ints.
    """
    if not isinstance(chunk, tuple) or not chunk:
        raise ValueError(
            f"checkpoint lengths must be a non-empty tuple of ints, got {chunk!r}"
        )
    for length in chunk:
        if (isinstance(length, bool) or not isinstance(length, numbers.Integral)
                or length <= 0):
            raise ValueError(
                f"every checkpoint length must be a positive int, got {length!r} "
                f"in {chunk!r}"
            )
    lengths = tuple(int(length) for length in chunk)
    for outer, inner in zip(lengths, lengths[1:]):
        if not inner < outer:
            raise ValueError(
                f"checkpoint lengths must be strictly decreasing, got {inner} "
                f"after {outer} in {chunk!r}"
            )
    return lengths


def segmented_scan(step_fn, carry, xs, chunk, scan=jax.lax.scan):
    """Scan a pytree record in checkpointed chunks and a real partial tail.

    ``chunk`` is None, an int (one level), or a tuple of step counts,
    outermost first: ``(444, 12)`` scans blocks of 444 steps, each made of
    checkpointed segments of 12. At every level the length is clamped by
    :func:`effective_chunk` for the sub-record it splits (a level the clamp
    turns off is skipped), and a partial tail is one more checkpointed unit
    that the inner levels split again. The record is never padded.

    ``scan`` controls the outer scan, allowing callers to retain recording
    hooks. Inner scans always use ``jax.lax.scan``. Inactive segmentation
    delegates directly to ``scan`` without adding a checkpoint wrapper.
    """
    if isinstance(chunk, tuple):
        return _nested_scan(step_fn, carry, xs, checkpoint_lengths(chunk), scan)
    n = jax.tree_util.tree_leaves(xs)[0].shape[0]
    chunk = effective_chunk(n, chunk)
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


def _nested_scan(step_fn, carry, xs, lengths, scan):
    """One level of :func:`segmented_scan` for validated ``lengths``."""
    if not lengths:
        return scan(step_fn, carry, xs)
    n = jax.tree_util.tree_leaves(xs)[0].shape[0]
    chunk, rest = effective_chunk(n, lengths[0]), lengths[1:]
    if chunk is None:
        return _nested_scan(step_fn, carry, xs, rest, scan)

    m, r = divmod(n, chunk)
    end = m * chunk
    segments = jax.tree_util.tree_map(
        lambda x: x[:end].reshape((m, chunk) + x.shape[1:]), xs
    )

    def unit_body(c, s):
        return _nested_scan(step_fn, c, s, rest, jax.lax.scan)

    body = jax.checkpoint(unit_body)
    carry, ys = scan(body, carry, segments)
    ys = jax.tree_util.tree_map(
        lambda y: y.reshape((end,) + y.shape[2:]), ys
    )
    if r:
        # The tail is one more checkpointed unit of r steps through the
        # caller's ``scan``; inside it the inner levels split the r steps.
        tail = jax.tree_util.tree_map(lambda x: x[end:][None], xs)
        carry, tail_ys = scan(body, carry, tail)
        ys = jax.tree_util.tree_map(
            lambda y, t: jnp.concatenate((y, t[0]), axis=0), ys, tail_ys
        )
    return carry, ys


class CheckpointMemory(NamedTuple):
    """The memory model of a checkpointed reverse pass, in bytes.

    ``peak_bytes = static_bytes + boundary_bytes * held_boundaries
    + step_bytes * live_steps + (cotangents - 1) * (working_bytes + boundary_bytes)``.
    ``held_boundaries`` is the number of
    boundary carries held at once: the units of the outermost level plus,
    for every inner level, the units per enclosing unit (tail units
    included). ``live_steps`` is the innermost length, the per-step record
    of the unit being reversed. ``lengths`` are the lengths after the clamp
    of :func:`effective_chunk` (None when no level is active, which keeps
    the whole per-step record). It is an estimate, not a bound.
    """

    n: int
    lengths: tuple | None
    levels: int
    held_boundaries: int
    live_steps: int
    static_bytes: float
    boundary_bytes: float
    step_bytes: float
    peak_bytes: float
    forward_passes: int


def validate_cotangents(cotangents):
    """Require a positive integer cotangent count, excluding bools."""
    if (isinstance(cotangents, bool) or not isinstance(cotangents, numbers.Integral)
            or cotangents <= 0):
        raise ValueError(f"cotangents must be a positive int, got {cotangents!r}")
    return int(cotangents)


def checkpoint_memory(n, lengths, *, boundary_bytes, step_bytes, static_bytes,
                      cotangents: int = 1, working_bytes=0):
    """Evaluate the memory model for ``n`` steps and ``lengths``.

    ``lengths`` is None, an int or a tuple as for :func:`segmented_scan`;
    each level is clamped exactly as the scan clamps its full units.
    Each extra cotangent adds ``working_bytes + boundary_bytes``. The
    caller supplies c (static_bytes) and w (working_bytes) from its scene.
    """
    cotangents = validate_cotangents(cotangents)
    if lengths is None:
        requested = ()
    elif isinstance(lengths, tuple):
        requested = checkpoint_lengths(lengths)
    else:
        requested = (lengths,)
    active = active_levels(n, requested)
    held, enclosing = 0, n
    for used in active:
        held += math.ceil(enclosing / used)
        enclosing = used
    live = active[-1] if active else n
    peak = (static_bytes + boundary_bytes * held + step_bytes * live
            + (cotangents - 1) * (working_bytes + boundary_bytes))
    return CheckpointMemory(
        n=int(n), lengths=tuple(active) if active else None, levels=len(active),
        held_boundaries=held, live_steps=live, static_bytes=static_bytes,
        boundary_bytes=boundary_bytes, step_bytes=step_bytes, peak_bytes=peak,
        forward_passes=1 + len(active),
    )


class CheckpointPlan(NamedTuple):
    """Checkpoint lengths chosen for a memory budget (see the planner)."""

    lengths: tuple | None
    levels: int
    peak_bytes: float
    forward_passes: int
    fits: bool
    budget_bytes: float
    memory: CheckpointMemory


def _candidates(n, k):
    """Valid length tuples with k levels and equal units per inner level."""
    if k == 1:
        for length in range(1, n // 2 + 1):
            if effective_chunk(n, length) == length:
                yield (length,)
        return
    for innermost in range(1, n // 2 + 1):
        ideal = (n / innermost) ** (1.0 / k)
        for units in sorted({math.floor(ideal), math.ceil(ideal)}):
            if units < 2:
                continue
            lengths = tuple(innermost * units ** (k - 1 - j) for j in range(k))
            if effective_chunk(n, lengths[0]) == lengths[0]:
                yield lengths


def plan_checkpoint_lengths(n, *, boundary_bytes, step_bytes, static_bytes,
                            budget_bytes, max_levels=3, cotangents: int = 1, working_bytes=0):
    """Choose checkpoint lengths for ``n`` steps under ``budget_bytes``.

    For k = 1, 2, ..., ``max_levels`` the lengths minimising the model of
    :func:`checkpoint_memory` with k levels are found (a search over the
    innermost length, the units per level as equal as integers allow), and
    the first k whose minimum fits the budget is returned: fewest levels
    first, because each level adds one forward pass of the record. Every
    returned length survives the clamp of :func:`effective_chunk` unchanged,
    so the model describes the scan that runs. When no k fits, ``fits`` is
    False and the plan with the smallest modelled peak is returned. The plan
    depends only on its arguments.
    ``cotangents`` counts simultaneous pullbacks; each extra one adds
    ``working_bytes + boundary_bytes``.
    """
    cotangents = validate_cotangents(cotangents)
    n = int(n)
    max_levels = int(max_levels)
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    if max_levels < 1:
        raise ValueError(f"max_levels must be >= 1, got {max_levels}")
    bytes_ = dict(boundary_bytes=boundary_bytes, step_bytes=step_bytes,
                  static_bytes=static_bytes, cotangents=cotangents, working_bytes=working_bytes)
    best_overall = None
    for k in range(1, max_levels + 1):
        best = None
        for lengths in _candidates(n, k):
            memory = checkpoint_memory(n, lengths, **bytes_)
            if memory.levels != k:
                continue
            if best is None or memory.peak_bytes < best.peak_bytes:
                best = memory
        if best is None:
            continue
        if best.peak_bytes <= budget_bytes:
            return _plan(best, True, budget_bytes)
        if best_overall is None or best.peak_bytes < best_overall.peak_bytes:
            best_overall = best
    if best_overall is None:
        best_overall = checkpoint_memory(n, None, **bytes_)
    return _plan(best_overall, best_overall.peak_bytes <= budget_bytes,
                 budget_bytes)


def _plan(memory, fits, budget_bytes):
    return CheckpointPlan(
        lengths=memory.lengths, levels=memory.levels,
        peak_bytes=memory.peak_bytes, forward_passes=memory.forward_passes,
        fits=fits, budget_bytes=budget_bytes, memory=memory,
    )
