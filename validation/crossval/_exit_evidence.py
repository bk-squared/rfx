"""Make a crossval record's persisted exit code the code the process returns.

Issue #946. Every crossval script that retains a record (#937) decides its
exit code once and writes it into the record BEFORE the run is over, on
purpose: printing is not persisting, and the record has to survive a crash in
the optional plotting stage that follows it. The cost of writing early is that
the record states an outcome the process has not reached yet. Whoever later
adds an exit path BETWEEN the write and the tail -- the abandoned #907 branch
added ``if PASS and verdict.q_vacuous: sys.exit(2)`` after cv02's writer --
makes the retained record disagree with the run: the process reported
inconclusive, the committed file still claimed ``exit_code 0`` and
``ALL CHECKS PASSED``. Nothing failed, because nothing compared the two.

This module keeps the early write and closes the gap behind it.
``write_record()`` persists the record now and ARMS a finalizer. If the
process then ends with a different status, the finalizer amends the persisted
record: ``exit_code`` becomes the status the process actually returned, the
declared value and summary are kept beside it under ``exit_code_declared`` /
``summary_declared``, and ``exit_code_reconciliation`` says what happened and
how it was observed. When the process ends with the code the record declared
-- the normal case, and the only one on ``main`` today -- the finalizer writes
nothing at all and the file stays byte for byte what the run produced. That
matters for the committed cv01/cv02 records, whose manifest entries claim a
re-run reproduces them bit-identically.

What the finalizer can see:

===========================  ==================================================
``sys.exit(code)``           a wrapper installed on ``sys.exit`` records it
uncaught exception           a ``sys.excepthook`` wrapper records status 1
normal completion            status 0
===========================  ==================================================

Two paths no in-process mechanism can see, stated rather than implied:
``os._exit()`` and a fatal signal skip ``atexit`` entirely; and a bare
``raise SystemExit(n)`` is invisible to the wrapper, because CPython handles
SystemExit before ``sys.excepthook`` is consulted. The second is closed by
contract, not by this module: ``tests/contracts/test_crossval_exit_code_evidence.py``
requires every script that writes an exit-code record to leave through
``sys.exit`` and to route the write through ``write_record``.

WHEN THE FINALIZER ARMS. Only when ``write_record``'s caller is running as
the program -- its module ``__name__`` is ``"__main__"``. A test that imports
a case and calls its ``main()`` gets the record written and nothing armed,
because the pytest process's exit status is not that case's verdict, and a
``pytest.raises(SystemExit)`` in an unrelated test would otherwise be read as
this run's outcome. The consequence for a future case: call ``write_record``
from the case script's own body, not from a helper module it imports, or the
record is persisted unarmed.

Usage -- the caller no longer holds a second copy of the code or the summary,
because ``write_record`` is what puts both into the document:

    import _exit_evidence

    rc = _exit_evidence.write_record(
        artifact_path, doc,
        exit_code=_exit_code(rfx_ok, have_meep, judged_ok),
        summary=lambda code: "ALL CHECKS PASSED" if code == 0 else "...",
    )
    ...
    sys.exit(rc)
"""

from __future__ import annotations

import atexit
import json
import os
import sys
from typing import Any, Callable

#: Key of the verdict block inside a crossval record.
VERDICT_KEY = "verdict"
EXIT_CODE_KEY = "exit_code"
SUMMARY_KEY = "summary"
#: Written only when the run's actual status differed from the declared one.
DECLARED_EXIT_CODE_KEY = "exit_code_declared"
DECLARED_SUMMARY_KEY = "summary_declared"
RECONCILIATION_KEY = "exit_code_reconciliation"

_RECONCILED_NOTE = (
    "the process exited with a status different from the one this record "
    "declared when it was written; exit_code is the status the process "
    "actually returned and exit_code_declared is what the verdict stage "
    "decided (issue #946)"
)


class _Armed:
    """One persisted record whose exit code is still open."""

    __slots__ = ("path", "verdict_key", "declared", "summary", "indent")

    def __init__(self, path: str, verdict_key: str, declared: int,
                 summary: Any, indent: int) -> None:
        self.path = path
        self.verdict_key = verdict_key
        self.declared = declared
        self.summary = summary
        self.indent = indent


# Keyed by absolute path: a script that rewrites the same record keeps one arm.
_armed: "dict[str, _Armed]" = {}
_observed_code: "int | None" = None
_observed_via: str = "normal completion"
_installed = False


def normalize_exit_code(code: Any) -> int:
    """The status a POSIX parent sees for ``sys.exit(code)``.

    ``None`` is 0, a string (or any other object) is 1 -- CPython prints it
    and exits 1 -- and an int is taken modulo 256, which is what ``waitpid``
    reports and therefore what a subprocess return code can be compared to.
    """
    if code is None:
        return 0
    if isinstance(code, bool):
        return int(code)
    if isinstance(code, int):
        return code % 256
    return 1


def _observe(code: Any, via: str) -> None:
    global _observed_code, _observed_via
    _observed_code = normalize_exit_code(code)
    _observed_via = via


def _install() -> None:
    """Wrap ``sys.exit`` / ``sys.excepthook`` and register the finalizer once.

    Once per module instance, not once per process: both wrappers chain to
    whatever was there before, so a second copy of this module (loaded under
    another name) observes the same exits instead of silently holding records
    it can never amend.
    """
    global _installed
    if _installed:
        return
    _installed = True

    previous_exit = sys.exit

    def _tracking_exit(code=None):  # noqa: ANN001 - mirrors sys.exit
        _observe(code, "sys.exit")
        previous_exit(code)

    sys.exit = _tracking_exit  # type: ignore[assignment]

    previous_hook = sys.excepthook

    def _tracking_excepthook(exc_type, exc, tb):  # noqa: ANN001
        # SystemExit never reaches an excepthook -- CPython handles it first --
        # so the isinstance branch is only for a hook chained ahead of ours.
        if isinstance(exc, SystemExit):
            _observe(exc.code, "SystemExit")
        else:
            _observe(1, "uncaught %s" % exc_type.__name__)
        previous_hook(exc_type, exc, tb)

    sys.excepthook = _tracking_excepthook
    atexit.register(_finalize)


def write_record(path: str, doc: dict, *, exit_code: Any,
                 summary: "str | Callable[[int], str] | None" = None,
                 verdict_key: str = VERDICT_KEY, indent: int = 1,
                 arm: "bool | None" = None) -> int:
    """Persist ``doc`` at ``path`` now; amend its exit code if the run ends
    with a different status.

    ``exit_code`` is written into ``doc[verdict_key]["exit_code"]`` by this
    function, so the record's field and the value the script exits with cannot
    be two copies that drift. ``summary`` may be a string (stored as-is) or a
    callable taking the exit code, in which case it is also what regenerates
    the summary if the record has to be amended.

    ``arm`` defaults to "the caller is running as the program" -- see the
    module docstring; pass it explicitly only in a test of this module.

    Returns the normalized exit code, for the caller to ``sys.exit()`` or
    ``return``.
    """
    if arm is None:
        arm = sys._getframe(1).f_globals.get("__name__") == "__main__"
    code = normalize_exit_code(exit_code)
    verdict = doc.setdefault(verdict_key, {})
    if not isinstance(verdict, dict):
        raise TypeError(
            "%s[%r] must be a dict to carry the verdict, got %r"
            % ("record", verdict_key, type(verdict).__name__))
    verdict[EXIT_CODE_KEY] = code
    if summary is not None:
        verdict[SUMMARY_KEY] = summary(code) if callable(summary) else summary

    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w") as handle:
        json.dump(doc, handle, indent=indent)

    if arm:
        _install()
        _armed[os.path.abspath(path)] = _Armed(
            os.path.abspath(path), verdict_key, code, summary, indent)
    return code


def armed_records() -> "dict[str, int]":
    """{absolute record path: declared exit code} -- for tests and debugging."""
    return {path: arm.declared for path, arm in _armed.items()}


def _warn(message: str) -> None:
    try:
        sys.stderr.write(message + "\n")
        sys.stderr.flush()
    except Exception:  # pragma: no cover - stderr closed during shutdown
        pass


def _reconcile(arm: _Armed, actual: int, via: str) -> None:
    try:
        with open(arm.path) as handle:
            doc = json.load(handle)
    except Exception as exc:  # pragma: no cover - unreadable record
        _warn("EXIT-CODE RECONCILIATION FAILED: %s declares exit %d but the "
              "process is exiting %d, and the record could not be re-read "
              "(%s). The retained record is WRONG."
              % (arm.path, arm.declared, actual, exc))
        return

    verdict = doc.get(arm.verdict_key)
    if not isinstance(verdict, dict):
        verdict = {}
        doc[arm.verdict_key] = verdict
    if SUMMARY_KEY in verdict:
        verdict[DECLARED_SUMMARY_KEY] = verdict[SUMMARY_KEY]
    verdict[DECLARED_EXIT_CODE_KEY] = arm.declared
    verdict[EXIT_CODE_KEY] = actual
    if callable(arm.summary):
        verdict[SUMMARY_KEY] = arm.summary(actual)
    else:
        verdict[SUMMARY_KEY] = (
            "EXIT %d -- the verdict stage declared exit %d, the process "
            "returned %d" % (actual, arm.declared, actual))
    verdict[RECONCILIATION_KEY] = {
        "declared": arm.declared,
        "actual": actual,
        "observed_via": via,
        "note": _RECONCILED_NOTE,
    }

    try:
        with open(arm.path, "w") as handle:
            json.dump(doc, handle, indent=arm.indent)
    except Exception as exc:  # pragma: no cover - unwritable record
        _warn("EXIT-CODE RECONCILIATION FAILED: could not rewrite %s (%s). "
              "It still claims exit %d; the process is exiting %d."
              % (arm.path, exc, arm.declared, actual))
        return

    _warn("EXIT-CODE RECONCILED: %s declared exit %d, the process is exiting "
          "%d (%s). The record now carries %d; the declared verdict is kept "
          "under %r / %r."
          % (arm.path, arm.declared, actual, via, actual,
             DECLARED_EXIT_CODE_KEY, DECLARED_SUMMARY_KEY))


def _finalize() -> None:
    actual = 0 if _observed_code is None else _observed_code
    via = _observed_via
    for arm in list(_armed.values()):
        if arm.declared == actual:
            continue
        _reconcile(arm, actual, via)


def _reset_for_tests() -> None:
    """Disarm every record and forget the observed status (unit tests only)."""
    global _observed_code, _observed_via
    _armed.clear()
    _observed_code = None
    _observed_via = "normal completion"
