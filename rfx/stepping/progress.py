"""Host-side progress lines and chunk assembly for the time-step loops.

Moved from ``rfx.progress``, which re-exports every name here.
"""

from __future__ import annotations

import time

from rfx.core.jax_utils import is_tracer

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

    def note(self, text: str) -> str:
        """Emit one line that is not a progress tick: how the run ended
        (``until_decay``'s stop criterion). It does not count as a report."""
        head = f"  [STOP] {self.label}: " if self.label else "  [STOP] "
        line = f"{head}{text}"
        print(line, file=self._stream, flush=True)
        return line
