"""ForwardResult assembly shared by the two nonuniform API lanes."""


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
