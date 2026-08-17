"""Host-side progress reporting for long solves (issue #667).

A single rfx solve can run for hours and, before this module existed, printed
nothing between the call and its return, so a slow run was indistinguishable
from a hang. The measured case that motivated it: a 42.15 M-cell /
225,000-step ``compute_msl_s_matrix`` whose last log line was written at
second 0 and was still the last line 4 h 10 min later.

Two solve shapes need two mechanisms:

* :func:`rfx.simulation.run_until_decay` is already a Python loop over a
  jitted single step, so it only needs a tick inside the loop
  (:class:`ProgressReporter`).
* :func:`rfx.simulation.run` is a single ``jax.lax.scan``. Printing from
  inside the scan body means ``jax.debug.print`` / ``io_callback`` on the
  hot path, which is exactly what the issue does not ask for. Instead
  :func:`scan_with_progress` splits the step range on the HOST and calls the
  same compiled scan once per chunk, threading ``carry`` through. This is a
  continuation, not a re-solve: the carry already holds the DFT
  accumulators and every port / flux / monitor state, so chunk re-entry
  contributes nothing numerically.

  That claim is not assumed. It is the same architecture the non-uniform
  ``until_decay`` lane already shipped (issue #383), whose chunk re-entry is
  locked at exact equality, and it is re-locked here for the uniform lane by
  ``tests/test_run_progress_reporting.py`` with SHA-256 digests over raw
  field bytes, the probe time series, the DFT-plane accumulators and the
  extracted S / Z0 / beta.

Design rules this module obeys:

* **Nothing branches on a traced value.** Every loop bound and comparison
  here is Python-int arithmetic on concrete values. Because wall-clock
  reporting is meaningless under tracing (a traced call would print once, at
  trace time, with a fabricated elapsed), :func:`check_not_traced` rejects
  the request loudly instead of silently emitting a wrong line.
* **Reporting must not change a result.** With ``report_every=None`` the
  caller runs the unchanged code path, so identity is by construction; with
  reporting on, the chunk loop is numerically a continuation.
* **Emit on preflight's channel.** ``rfx`` has no ``logging`` configuration;
  preflight writes ``  [PREFLIGHT] ...`` to stdout, so progress writes
  ``  [PROGRESS] ...`` to stdout, ``flush=True`` so a redirected multi-hour
  job log is not left in a stale buffer.
"""

from __future__ import annotations

import time

from rfx.core.jax_utils import is_tracer

__all__ = [
    "ProgressReporter",
    "check_not_traced",
    "scan_with_progress",
    "validate_report_every",
]

_PREFIX = "  [PROGRESS]"

_TRACED_MSG = (
    "report_every is a host-side progress feature and cannot run under "
    "jax.jit / jax.grad / jax.vmap: the chunk loop reads the host wall clock "
    "and prints, so a traced call would emit one line at TRACE time with a "
    "fabricated elapsed and rate. Drop report_every on the differentiable "
    "path (forward() / optimize() / eps_override=...), or call the eager "
    "entry point (run() / run_until_decay()) outside the transform."
)


def _fmt_hms(seconds: float) -> str:
    """Format a duration as ``H:MM:SS`` (``--:--:--`` when not finite)."""
    if not (seconds == seconds) or seconds in (float("inf"), float("-inf")):
        return "--:--:--"
    seconds = max(0.0, float(seconds))
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}"


def validate_report_every(report_every: object, *, n_steps: int) -> int:
    """Coerce and check ``report_every``; return it as a Python int.

    ``n_steps`` is used only for the advisory in the error message, never as
    an upper bound: a ``report_every`` larger than ``n_steps`` is legal and
    simply yields one report at the end.
    """
    if is_tracer(report_every):
        raise ValueError(_TRACED_MSG)
    try:
        every = int(report_every)
    except (TypeError, ValueError):
        raise ValueError(
            f"report_every must be an integer number of steps or None, got "
            f"{report_every!r}"
        ) from None
    if every != report_every:
        # Truncating 1000.5 -> 1000 would silently report on a cadence the
        # caller did not ask for.
        raise ValueError(
            f"report_every must be a whole number of steps, got "
            f"{report_every!r}"
        )
    if every < 1:
        raise ValueError(
            f"report_every must be >= 1 step, got {report_every!r} "
            f"(pass None to disable progress reporting; a run of "
            f"{n_steps} steps typically wants report_every around "
            f"{max(1, n_steps // 20)})"
        )
    return every


def check_not_traced(*trees: object) -> None:
    """Raise if any leaf of *trees* is a JAX tracer.

    Called before the host chunk loop so a ``report_every`` request made
    under ``jit``/``grad``/``vmap`` fails with an explanation instead of
    printing a trace-time line.
    """
    import jax

    for tree in trees:
        for leaf in jax.tree_util.tree_leaves(tree):
            if is_tracer(leaf):
                raise ValueError(_TRACED_MSG)


class ProgressReporter:
    """One line per report: steps done / total, elapsed, rate, ETA.

    Parameters
    ----------
    total_steps : int
        Denominator of the progress fraction. On the ``until_decay`` lane
        this is the ``max_steps`` cap rather than a known length; pass
        ``total_is_cap=True`` so the line says so and the ETA is read as an
        upper bound.
    label : str
        Short caller-supplied tag, e.g. ``"MSL drive p1"``. Distinguishes
        the per-drive solves of one ``compute_*_s_matrix`` call, which are
        otherwise identical lines.
    stream : file-like or None
        Destination; ``None`` means ``sys.stdout`` (``print``'s default).

    The wall clock starts at construction, so build the reporter
    immediately before the first step.
    """

    def __init__(
        self,
        total_steps: int,
        *,
        label: str = "",
        total_is_cap: bool = False,
        stream: object | None = None,
    ) -> None:
        self.total = int(total_steps)
        self.label = str(label)
        self.total_is_cap = bool(total_is_cap)
        self._stream = stream
        self._t0 = time.perf_counter()
        self.last_reported = 0

    def report(self, steps_done: int) -> str:
        """Emit one progress line for *steps_done* and return it.

        The elapsed time is measured from construction, so the caller is
        responsible for having synchronised the device first (see
        :func:`scan_with_progress`); otherwise the implied rate would be the
        host dispatch rate, not the solve rate.
        """
        steps_done = int(steps_done)
        elapsed = time.perf_counter() - self._t0
        rate = steps_done / elapsed if elapsed > 0.0 else float("inf")
        remaining = max(0, self.total - steps_done)
        eta = remaining / rate if rate > 0.0 else float("inf")
        pct = (100.0 * steps_done / self.total) if self.total > 0 else 100.0
        total_txt = f"{self.total}{' (cap)' if self.total_is_cap else ''}"
        head = f"{_PREFIX} {self.label}: " if self.label else f"{_PREFIX} "
        line = (
            f"{head}{steps_done}/{total_txt} steps ({pct:.1f}%) | "
            f"elapsed {_fmt_hms(elapsed)} | {rate:.1f} steps/s | "
            f"ETA {_fmt_hms(eta)}"
        )
        print(line, file=self._stream, flush=True)
        self.last_reported = steps_done
        return line


def scan_with_progress(
    body,
    carry_init,
    xs,
    *,
    n_steps: int,
    report_every: int,
    label: str = "",
    stream: object | None = None,
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

    Cost, so it is not buried: each report inserts a device
    synchronisation (without one the host would dispatch every chunk
    immediately and print all the lines at t = 0 with a fabricated rate),
    and the per-chunk ``outputs`` are concatenated once at the end, so peak
    memory transiently holds both the chunk pieces and the joined array.
    For the S-parameter runs this feature targets, ``outputs`` is a
    ``(n_steps, n_probes)`` time series that is negligible beside the field
    arrays; a snapshot-recording run pays the concatenation on the snapshot
    stack too, so prefer a large ``report_every`` there.
    """
    import jax
    import jax.numpy as jnp

    every = validate_report_every(report_every, n_steps=n_steps)
    check_not_traced(carry_init, xs)

    n_steps = int(n_steps)
    if n_steps <= 0:
        # Degenerate length: defer to the plain scan so the zero-length
        # output structure matches the unchunked path exactly.
        return jax.lax.scan(body, carry_init, xs)

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

    reporter = ProgressReporter(n_steps, label=label, stream=stream)
    carry = carry_init
    chunk_outputs = []
    done = 0
    while done < n_steps:
        this = min(every, n_steps - done)
        lo, hi = done, done + this
        xs_chunk = jax.tree_util.tree_map(lambda a: a[lo:hi], xs)
        carry, ys = jax.lax.scan(body, carry, xs_chunk)
        # Block before reading the clock: JAX dispatch is asynchronous, so
        # an unsynchronised loop would queue every chunk and report the
        # host's dispatch rate instead of the solver's.
        carry = jax.block_until_ready(carry)
        chunk_outputs.append(ys)
        done = hi
        reporter.report(done)

    if len(chunk_outputs) == 1:
        outputs = chunk_outputs[0]
    else:
        outputs = jax.tree_util.tree_map(
            lambda *parts: jnp.concatenate(parts, axis=0), *chunk_outputs
        )
    return carry, outputs
