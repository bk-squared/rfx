"""Host progress chunks around the canonical compiled slab step."""
from functools import partial

import jax.numpy as jnp

from rfx.stepping.progress_scan import scan_with_progress


def run_chunks(carry, invariants, waves, ranks, advance, *, n_steps,
               compile_loop, scan_loop):
    from rfx.stepping.sharded import recorded_sharded_call
    @compile_loop
    def run_fn(carry, invariants, xs, ranks):
        step = partial(advance, invariants=invariants, ranks=ranks)
        return scan_loop(step, carry, xs)

    def chunk_scan(carry, xs_chunk, lo):
        # xs carries absolute indices, including into persistent record arrays.
        return recorded_sharded_call(run_fn, carry, invariants, xs_chunk, ranks)

    xs = (jnp.arange(n_steps, dtype=jnp.int32), waves)
    return scan_with_progress(None, carry, xs, n_steps=n_steps,
                              report_every=None, inherit_cadence=False, chunk_scan=chunk_scan,
                              trace_probes=(invariants, ranks))
