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
  here is Python-int arithmetic on concrete values.
* **Tracing is rejected on the routes it actually arrives by, and the guard
  claims no more than that.** Wall-clock reporting is meaningless under a
  trace, so :func:`check_not_traced` raises when a tracer is found — but it
  is a check on the pytrees it is HANDED, not a general "is a trace active?"
  test, which JAX 0.10 does not expose. Callers pass the carry, the
  per-step ``xs`` AND the material / geometry arrays, which is how
  ``forward()`` / ``optimize()`` trace this code. A tracer arriving solely
  through some other closure escapes the guard and produces trace-time
  progress lines; measured, the computed values stay byte-identical in that
  case, so the leak is cosmetic. See :func:`check_not_traced`.
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
    "concat_chunks",
    "scan_with_progress",
    "validate_report_every",
]

_PREFIX = "  [PROGRESS]"

# Bounded argument count per concatenate call. `jnp.concatenate` takes one
# ARGUMENT per chunk, and XLA trace+compile cost grows superlinearly in the
# argument count -- which bites exactly the long runs this feature exists
# for. Measured on a (225000, 20) float32 assembly, first (compiling) call:
#
#     chunks   flat jnp.concatenate   grouped (this constant)
#      2,250              2.567 s                   0.133 s
#     22,500            232.2   s                   0.994 s
#
# Grouping is exact: concatenation is associative, so the joined bytes are
# unchanged (locked by test_grouped_concat_matches_flat_concat).
_CONCAT_GROUP = 64

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
    if isinstance(report_every, bool):
        # Python bools ARE ints, so True would silently mean "every step".
        raise ValueError(
            f"report_every must be an integer number of steps or None, got "
            f"{report_every!r} (a bool). Pass None to disable reporting."
        )
    try:
        every = int(report_every)
    except (TypeError, ValueError, OverflowError):
        # OverflowError is what int(float('inf')) raises.
        raise ValueError(
            f"report_every must be a finite integer number of steps or None, "
            f"got {report_every!r}"
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

    Scope, stated exactly, because a guard that is believed to cover more
    than it does is worse than a narrow one:

    **This inspects the trees it is handed, and nothing else.** It is not a
    general "is a JAX trace active?" test — JAX 0.10 exposes no such API
    (``jax.core.trace_state_clean`` does not exist), and a freshly built
    constant is a tracer only under ``jit``, not under bare ``grad`` or
    ``vmap``. So callers must hand it every input a tracer could realistically
    arrive on. :func:`rfx.simulation.run` passes the initial carry, the
    per-step ``xs``, and the material / geometry arrays, which is the route
    ``forward()`` and ``optimize()`` actually use.

    What escapes it: a tracer that reaches the scan body ONLY through a
    closure the caller did not pass here — e.g. hand-built coefficient
    arrays under a bare ``jax.grad`` around the low-level entry point. In
    that case progress lines are printed once, at trace time, with a
    meaningless elapsed and rate. **The computed result is unaffected** —
    reporting-on and reporting-off are byte-identical under ``grad`` and
    ``vmap`` — so this is a cosmetic leak, not a correctness hole.
    """
    import jax

    for tree in trees:
        for leaf in jax.tree_util.tree_leaves(tree):
            if is_tracer(leaf):
                raise ValueError(_TRACED_MSG)


def concat_chunks(chunk_outputs: list):
    """Join per-chunk scan outputs along the step axis, exactly.

    Uses bounded-arity grouping rather than one flat
    ``jnp.concatenate(*all_chunks)`` -- see :data:`_CONCAT_GROUP` for the
    measured reason. Concatenation is associative, so the result is
    bit-identical to the flat form; only the XLA compile cost differs.
    """
    import jax
    import jax.numpy as jnp

    if len(chunk_outputs) == 1:
        return chunk_outputs[0]

    parts = list(chunk_outputs)
    while len(parts) > 1:
        grouped = []
        for i in range(0, len(parts), _CONCAT_GROUP):
            group = parts[i:i + _CONCAT_GROUP]
            if len(group) == 1:
                grouped.append(group[0])
            else:
                grouped.append(jax.tree_util.tree_map(
                    lambda *ps: jnp.concatenate(ps, axis=0), *group))
        parts = grouped
    return parts[0]


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
    trace_probes: tuple = (),
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

    every = validate_report_every(report_every, n_steps=n_steps)
    check_not_traced(carry_init, xs, *trace_probes)

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

    return carry, concat_chunks(chunk_outputs)
