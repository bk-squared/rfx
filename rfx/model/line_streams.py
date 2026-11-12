"""The two serial streams of a smoothed sample's normal line.

The merge stream turns the painted pieces of a line into runs of one medium;
the own-run stream turns the own group's intervals into runs over the
support. ``rfx.geometry.outside_content`` calls both while it builds the
outside content of the smoothed permittivity, which is part of the realized
model; the streams are loops, so they live in the model layer.
"""

import jax
import numpy as np


# The two streams below are serial recurrences over the pieces of a line.
# Each has ONE step function that holds its rule; two drivers apply it in
# order. NumPy (the host path) calls it in a Python loop. A traced build
# runs the same step in ``jax.lax.scan``: an unrolled Python loop over
# full-grid arrays has no loop left in the compiled program, so XLA would
# inline the whole chain before each output into that output and compute
# it again, and the work would grow with the square of the piece count.
# The step is selects, comparisons and in-order sums, so both drivers give
# the same bits.


def _merge_step(xp, carry, piece):
    """One piece of the merge stream: start a run, extend it or close it.

    Empty sort slots cannot split a run. Returns the new carry and the
    run (lo, hi, eps) this piece emits.
    """
    start, end, value, code = carry
    low, high, piece_value, piece_code = piece
    positive = high > low
    change = positive & (end > start) & (piece_code != code)
    emitted = (start, xp.where(change, end, start), value)
    new = positive & ((end == start) | change)
    start = xp.where(new, low, start)
    value = xp.where(new, piece_value, value)
    code = xp.where(new, piece_code, code)
    end = xp.where(positive, high, end)
    return (start, end, value, code), emitted


def _merge_stream(xp, low, high, values, codes, scan=None):
    """Stream through the painted pieces; stacked (lo, hi, eps) runs.

    One run per piece, then the run still open after the last piece.
    ``scan`` picks the driver; by default ``jax.numpy`` scans and NumPy loops.
    """
    if scan is None:
        scan = xp is not np
    first = (low[0], low[0], values[0], codes[0])
    if not scan:
        carry, emitted = first, []
        for piece in zip(low, high, values, codes):
            carry, run = _merge_step(xp, carry, piece)
            emitted.append(run)
        start, end, value, _ = carry
        emitted.append((start, end, value))
        return tuple(xp.stack(column) for column in zip(*emitted))
    carry, emitted = jax.lax.scan(lambda c, piece: _merge_step(xp, c, piece), first,
                                  (low, high, values, codes))
    start, end, value, _ = carry
    return tuple(xp.concatenate([column, last[None]])
                 for column, last in zip(emitted, (start, end, value)))


def _own_stream(xp, lo, hi, flat, half, shape, scan=None):
    """Stream through the own intervals in lo order: (thin, inside, gaps).

    ``inside`` is the summed length of the runs inside the support.
    ``scan`` picks the driver; by default ``jax.numpy`` scans and NumPy loops.
    """
    if scan is None:
        scan = xp is not np
    carry = (xp.zeros(shape, dtype=bool), xp.zeros(shape, dtype=lo.dtype),
             xp.broadcast_to(-half, shape), lo[0], hi[0], flat[0])
    pieces = (lo[1:], hi[1:], flat[1:])
    if not scan:
        gaps = []
        for piece in zip(*pieces):
            carry, gap = _own_step(xp, carry, piece, half)
            gaps.append(gap)
    else:
        carry, (starts, ends) = jax.lax.scan(lambda c, piece: _own_step(xp, c, piece, half),
                                             carry, pieces)
        gaps = [(starts[i], ends[i]) for i in range(starts.shape[0])]
    # After the last interval the open run closes.
    carry, gap = _own_close(xp, carry, True, half)
    thin, inside, previous = carry[:3]
    gaps.append(gap)
    gaps.append((xp.maximum(previous, -half), xp.broadcast_to(half, shape)))
    return thin, inside, gaps


def _own_close(xp, carry, new, half):
    """Where ``new``, close the open run: mark it thin if it is planar with
    both ends inside the support, add its length inside the support and
    emit the gap before it."""
    thin, inside, previous, start, end, planar = carry
    thin = thin | (new & planar & (start > -half) & (end < half))
    held = xp.maximum(xp.minimum(end, half) - xp.maximum(start, -half), 0)
    inside = inside + xp.where(new, held, 0)
    gap = (xp.maximum(previous, -half), xp.where(new, xp.minimum(start, half), -half))
    previous = xp.where(new, end, previous)
    return (thin, inside, previous, start, end, planar), gap


def _own_step(xp, carry, piece, half):
    """One own interval, in lo order: it opens a new run where it starts
    past the open run's end (closing that run), else it extends the run."""
    lo, hi, flat = piece
    new = lo > carry[4]
    (thin, inside, previous, start, end, planar), gap = _own_close(xp, carry, new, half)
    start = xp.where(new, lo, start)
    planar = xp.where(new, flat, planar & flat)
    end = xp.where(new, hi, xp.maximum(end, hi))
    return (thin, inside, previous, start, end, planar), gap

