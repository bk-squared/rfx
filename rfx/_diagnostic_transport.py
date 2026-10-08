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
    return result if merged == existing else result._replace(diagnostics=merged)


def diagnostic_refusal(error, diagnostics=(), *, causes=None):
    """Attach preceding observations and known causes at the admission raise site."""
    cause = tuple(getattr(error, "diagnostics", ()) if causes is None else causes)
    if not any(d.severity == "refusal" for d in cause):
        cause += (from_legacy(error, refusal=True),)
    error.diagnostics = merge_diagnostics(diagnostics, cause)
    return error


def pack_nu_forward_result(
    *,
    diagnostics=(),
    time_series,
    grid,
    ntff_data=None,
    ntff_box=None,
    s_params=None,
    freqs=None,
    dft_planes=None,
    wire_port_sparams=None,
    sparam_time_records=None,
    dft_time_records=None,
    dt=None,
    current_moment_data=None,
    current_moment_monitor=None,
):
    """Assemble the minimal ``ForwardResult`` for both NU forward lanes.

    The single-device (:meth:`_forward_nonuniform_from_materials`) and
    distributed (:meth:`_forward_distributed_nonuniform_from_materials`)
    lanes are a hand-maintained mirror pair whose only shared concern is
    this final ``ForwardResult`` assembly. Each lane extracts its own
    per-lane field values (the single-device lane reads a ``Result``
    object; the distributed lane reads a runner dict and forces the
    unsupported observables to ``None``) and passes them here as explicit
    parameters — there is no closure capture of caller locals, matching
    the W6.1 ``_StepContext`` / W6.6 builder precedents. Centralising the
    constructor keeps the two lanes' output schema from drifting apart.

    ``wire_port_sparams`` is the per-port ``(meta, accs)`` pair tuple
    the single-device lane's ``Result`` carries; the distributed lane
    has no wire-port accumulators and leaves it ``None``.
    """
    from rfx.api._spec import ForwardResult
    return ForwardResult(
        diagnostics=diagnostics,
        time_series=time_series,
        ntff_data=ntff_data,
        ntff_box=ntff_box,
        grid=grid,
        s_params=s_params,
        freqs=freqs,
        dft_planes=dft_planes,
        wire_port_sparams=wire_port_sparams,
        sparam_time_records=sparam_time_records,
        dft_time_records=dft_time_records,
        dt=dt,
        current_moment_data=current_moment_data,
        current_moment_monitor=current_moment_monitor,
    )
