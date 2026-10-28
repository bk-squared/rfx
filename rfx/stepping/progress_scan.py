"""Host-chunked time stepping with progress reporting."""

from __future__ import annotations

from rfx.stepping.progress import ProgressReporter, check_not_traced, concat_chunks, validate_report_every


def scan_with_progress(
    body,
    carry_init,
    xs,
    *,
    n_steps: int,
    report_every: int,
    label: str = "",
    stream: object | None = None,
    trace_probes: tuple = (),
    chunk_scan=None,
    stop_fn=None,
    chunk: int | None = None,
):
    """``jax.lax.scan(body, carry_init, xs)`` split into host-side chunks.

    Returns the same ``(final_carry, stacked_outputs)`` pair as the
    equivalent single scan.

    Every chunk calls the SAME ``body`` with the SAME carry threaded
    through and with ``xs`` sliced on its leading axis, so the global step
    indices, source samples and DFT phases a chunk sees are exactly the ones
    the unchunked scan would have fed it at those steps. All full chunks
    share one XLA executable (identical shapes); a ragged final chunk
    compiles once more.

    ``trace_probes`` are extra pytrees checked for tracers alongside
    ``carry_init`` and ``xs``. Pass the material / geometry arrays here: a
    tracer entering the scan body through them is captured in ``body``'s
    closure and is otherwise invisible to :func:`check_not_traced` (see its
    docstring for exactly what that guard does and does not cover).

    ``chunk_scan``, when given, replaces the per-chunk
    ``jax.lax.scan(body, carry, xs_chunk)`` with
    ``chunk_scan(carry, xs_chunk, lo)``, ``lo`` being the global index of
    the chunk's first step. ``run()`` passes one that records field
    snapshots at global multiples of their interval (#1258); it must return
    the same ``(carry, outputs)`` pair, with every output leaf stacked on a
    leading axis that concatenates across chunks.

    ``stop_fn``, when given (issue #1254, ``run(..., until_identified=True)``),
    is called after every chunk as ``stop_fn(done, carry, chunk_outputs)``,
    ``done`` being the steps run so far and ``carry`` the state after them;
    a true return ends the loop there, and the returned outputs hold those
    ``done`` steps. The chunks are then ``chunk`` steps long, and
    ``report_every`` may be ``None`` (no progress lines) or a cadence,
    reported at the first chunk boundary at or past each interval and at the
    stop. With ``stop_fn=None`` the loop is the one above, unchanged.

    Costs, so they are not buried:

    * Each report inserts a device synchronisation. Without one the host
      would dispatch every chunk immediately and print all the lines at
      t = 0 with a fabricated rate.
    * The per-chunk ``outputs`` are joined once at the end, so peak memory
      transiently holds both the chunk pieces and the joined array. For the
      S-parameter runs this targets, ``outputs`` is a ``(n_steps, n_probes)``
      time series that is negligible beside the field arrays; a
      snapshot-recording run pays it on the snapshot stack too.
    * The join is grouped rather than one flat ``jnp.concatenate`` over
      every chunk, because a flat concatenate takes one ARGUMENT per chunk
      and XLA's compile cost grows superlinearly in argument count — 232 s
      at 22,500 chunks, against 0.99 s grouped. See :func:`concat_chunks`.
    """
    import jax

    if stop_fn is None:
        every = validate_report_every(report_every, n_steps=n_steps)
        step = every
    else:
        every = (None if report_every is None
                 else validate_report_every(report_every, n_steps=n_steps))
        if isinstance(chunk, bool) or not isinstance(chunk, int) or chunk < 1:
            raise ValueError(
                f"scan_with_progress(stop_fn=...) needs chunk= a positive "
                f"number of steps, got {chunk!r}")
        step = int(chunk)
    check_not_traced(carry_init, xs, *trace_probes)

    if chunk_scan is None:
        def chunk_scan(carry, xs_chunk, lo):
            return jax.lax.scan(body, carry, xs_chunk)

    n_steps = int(n_steps)
    if n_steps <= 0:
        # Degenerate length: defer to the plain scan so the zero-length
        # output structure matches the unchunked path exactly.
        return chunk_scan(carry_init, xs, 0)

    # The chunk loop slices every xs leaf on its leading axis, which is only
    # equivalent to the unchunked scan if that axis IS the step axis. A
    # mismatched leaf would silently feed the wrong samples rather than
    # fail, so check it here instead of trusting the caller.
    bad = [(i, leaf.shape) for i, leaf in
           enumerate(jax.tree_util.tree_leaves(xs))
           if leaf.shape[0] != n_steps]
    if bad:
        raise ValueError(
            f"scan_with_progress requires every xs leaf to have leading "
            f"dimension n_steps={n_steps}; got mismatched leaves "
            f"(index, shape): {bad}"
        )

    reporter = (None if every is None else
                ProgressReporter(n_steps, label=label, stream=stream,
                                 total_is_cap=stop_fn is not None))
    carry = carry_init
    chunk_outputs = []
    done = 0
    while done < n_steps:
        this = min(step, n_steps - done)
        if stop_fn is not None and n_steps - done - this == 1:
            # never a one-step last chunk: XLA inlines a loop that runs once,
            # and the inlined step can move the last bit of a field against
            # the single scan (rfx-known-issues, "A one-step scan compiles
            # differently"); the stopped record must be the single scan's.
            this += 1
        lo, hi = done, done + this
        xs_chunk = jax.tree_util.tree_map(lambda a: a[lo:hi], xs)
        carry, ys = chunk_scan(carry, xs_chunk, lo)
        # Block before reading the clock: JAX dispatch is asynchronous, so
        # an unsynchronised loop would queue every chunk and report the
        # host's dispatch rate instead of the solver's.
        carry = jax.block_until_ready(carry)
        chunk_outputs.append(ys)
        done = hi
        if reporter is not None and (
                stop_fn is None or done - reporter.last_reported >= every
                or done >= n_steps):
            reporter.report(done)
        if stop_fn is not None and stop_fn(done, carry, chunk_outputs):
            break

    if (stop_fn is not None and reporter is not None
            and reporter.last_reported != done):
        reporter.report(done)
    return carry, concat_chunks(chunk_outputs)
