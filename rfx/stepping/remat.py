"""Segment-level rematerialization without extending the time record."""

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


def segmented_scan(step_fn, carry, xs, chunk, scan=jax.lax.scan):
    """Scan a pytree record in checkpointed chunks and a real partial tail.

    ``scan`` controls the outer scan, allowing callers to retain recording
    hooks. Inner scans always use ``jax.lax.scan``. Inactive segmentation
    delegates directly to ``scan`` without adding a checkpoint wrapper.
    """
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
