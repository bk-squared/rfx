"""Retain a cv02 verdict before its optional tail and reconcile that tail.

This lexical scope owns only the case's tail, not the embedding interpreter:
there are no atexit handlers or global sys.exit hooks. A host that catches the
case's SystemExit keeps its own outcome. SIGKILL/os._exit cannot be reconciled.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import json
import os
from pathlib import Path
import signal
import sys
import tempfile


def _write(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as out:
        temp = Path(out.name)
        try:
            json.dump(record, out, indent=1)
            out.flush()
            os.replace(temp, path)
        finally:
            temp.unlink(missing_ok=True)


@contextmanager
def retained_verdict(path, document: dict):
    """Write before yielding; keep the case's declared and actual exits distinct."""
    path = Path(path)
    record = deepcopy(document)
    _write(path, record)

    def reconcile(code: int, reason: str) -> None:
        verdict = record["verdict"]
        if code == verdict["exit_code"]:
            return
        verdict["declared_exit_code"] = verdict["exit_code"]
        verdict["declared_summary"] = verdict["summary"]
        verdict["exit_code"] = code
        # A late exit must never manufacture a passing scientific verdict,
        # even when the tail accidentally returns zero after a failed judge.
        verdict["summary"] = f"PROCESS OUTCOME CHANGED AFTER VERDICT (exit {code})"
        verdict["process_outcome_reason"] = reason
        _write(path, record)

    try:
        yield
    except SystemExit as exc:
        # Python treats None as success and non-integer messages as exit 1.
        code = 0 if exc.code is None else (exc.code if isinstance(exc.code, int) else 1)
        # This runner is POSIX; the process status carries eight exit bits.
        # CPython's C-long conversion of an overflowing integer returns -1.
        code = (code if -sys.maxsize - 1 <= code <= sys.maxsize else -1) & 255
        reconcile(code, "SystemExit in the case tail")
        raise
    except KeyboardInterrupt:
        reconcile(-signal.SIGINT, "KeyboardInterrupt in the case tail")
        raise
    except Exception as exc:
        reconcile(1, f"unhandled {type(exc).__name__} in the case tail")
        raise
    else:
        reconcile(0, "normal return from the case tail")
