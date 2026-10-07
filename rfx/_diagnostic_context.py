"""Invocation-local collection and the single result/refusal assembly boundary."""

from contextlib import contextmanager
from contextvars import ContextVar
import warnings
from threading import RLock

import jax

from rfx.diagnostic_records import Diagnostic, from_legacy


# Concrete host metadata must not become numeric leaves under jit/grad/vmap.
jax.tree_util.register_static(Diagnostic)
_ACTIVE = ContextVar("rfx_diagnostic_scope", default=None)
_WARNING_CAPTURE_LOCK = RLock()


def record_diagnostics(diagnostics):
    current = _ACTIVE.get()
    if current is not None:
        for diagnostic in diagnostics:
            if diagnostic not in current:
                current.append(diagnostic)


class diagnostic_scope:
    """No wrapper frame around execution, preserving warning stack levels.

    Nested solves contribute the union of their records to the calculator.
    Deduplication compares the entire immutable record, not just its code:
    different ports, values, messages, paths or sources remain distinguishable.
    """

    def __init__(self, *, serialize_warnings=False):
        # Python 3.11 warning filters/capture are process-global. Public host
        # entry scopes serialize their setup and dispatch, including nested
        # calculator drives. Device work may still execute asynchronously.
        self.serialize_warnings = serialize_warnings

    def __enter__(self):
        if self.serialize_warnings:
            _WARNING_CAPTURE_LOCK.acquire()
        self.records = []
        self.token = _ACTIVE.set(self.records)
        return self

    def __exit__(self, kind, error, traceback):
        try:
            if isinstance(error, (ValueError, NotImplementedError)):
                cause = tuple(getattr(error, "diagnostics", ()))
                record_diagnostics(cause)
                if not any(d.severity == "refusal" for d in cause):
                    record_diagnostics(
                        (
                            from_legacy(
                                error,
                                code=getattr(error, "code", "uncoded"),
                                source=getattr(error, "source", None),
                                refusal=True,
                            ),
                        )
                    )
                error.diagnostics = tuple(self.records)
        finally:
            try:
                _ACTIVE.reset(self.token)
                record_diagnostics(self.records)
            finally:
                if self.serialize_warnings:
                    _WARNING_CAPTURE_LOCK.release()
        return False


def diagnostic_result(result, *inspection):
    """Pass the assembled tuple to the returned result, preserving other fields."""
    if inspection:
        return (diagnostic_result(result), *inspection)
    # Mixed's optional phasor inspection return is (result, inspection_dict).
    if isinstance(result, tuple) and not hasattr(result, "_fields"):
        return tuple(diagnostic_result(item) for item in result)
    if not hasattr(result, "diagnostics"):
        return result
    record_diagnostics(result.diagnostics)
    diagnostics = tuple(_ACTIVE.get() or result.diagnostics)
    if hasattr(result, "_replace"):
        return result._replace(diagnostics=diagnostics)
    result.diagnostics = diagnostics
    return result


@contextmanager
def capture_warnings(**kwargs):
    """Protect standalone report collection using the invocation warning lock."""
    with _WARNING_CAPTURE_LOCK, warnings.catch_warnings(**kwargs) as caught:
        yield caught
