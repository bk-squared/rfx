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
  ``tests/unit/runners/test_run_progress_reporting.py`` with SHA-256 digests over raw
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

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rfx.stepping.progress_scan import scan_with_progress

from rfx.stepping.progress import (
    _CONCAT_GROUP, _PREFIX, _TRACED_MSG, ProgressReporter, _fmt_hms, check_not_traced,
    concat_chunks, validate_report_every,
)

__all__ = [
    "ProgressReporter",
    "check_not_traced",
    "concat_chunks",
    "scan_with_progress",
    "validate_report_every",
]


# Lazy compatibility exports avoid a cycle with the moved implementation.
_LAZY_EXPORTS = {
    "scan_with_progress": "rfx.stepping.progress_scan",
}


def __getattr__(name):
    from importlib import import_module

    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(_LAZY_EXPORTS[name]), name)
    globals()[name] = value
    return value
