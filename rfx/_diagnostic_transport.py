"""Explicit diagnostic tuple transport; no invocation state or warning capture."""

import jax

from rfx.diagnostic_records import Diagnostic, from_legacy


jax.tree_util.register_static(Diagnostic)


def merge_diagnostics(*groups):
    """A calculator carries its own preflight and every solve's preflight records.

    Merge in observation order, deduplicating by whole-record equality; records
    with different subjects, values, messages, paths or sources remain distinct.
    Entries with no preflight contribute an empty tuple. Nothing is stored on a
    Simulation, in thread/context state, or in a process-global collection.
    """
    records = []
    for group in groups:
        for diagnostic in group:
            if diagnostic not in records:
                records.append(diagnostic)
    return tuple(records)


def report_diagnostics(report):
    """Read a report, including legacy test doubles that return no report."""
    return tuple(getattr(report, "diagnostics", ()))


def result_with_diagnostics(result, diagnostics):
    """Merge a fallback's parent and child reports; unchanged records are a no-op."""
    existing = report_diagnostics(result)
    merged = merge_diagnostics(diagnostics, existing)
    return (
        result
        if merged == existing
        else result._replace(
            diagnostics=merged,
        )
    )


def diagnostic_refusal(error, diagnostics=(), *, causes=None):
    """Attach preceding observations and known causes at the admission raise site."""
    cause = tuple(getattr(error, "diagnostics", ()) if causes is None else causes)
    if not any(d.severity == "refusal" for d in cause):
        cause += (
            from_legacy(
                error,
                refusal=True,
            ),
        )
    error.diagnostics = merge_diagnostics(diagnostics, cause)
    return error
