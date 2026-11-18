"""Completed probe spectra divided by the closed-box plane wave's incident spectrum.

With ``RingdownSpec(probe_normalization="incident")`` every completed probe
channel X_c(f) is returned as X_c(f) / I_c(f). I_c is the incident E the
plane wave carries at the channel's realized E-node plane x_c:

    I_c(f) = dt sum_{n=0}^{N-1} i_c(n) exp(-j 2 pi f (n + 1) dt),

with i_c(n) the 1-D incident line's E along the polarization unit vector
e_hat at x_c after the E update of step n, over the run's N steps. The stamp
(n + 1) dt is the E channel's own, so the quotient carries no clock offset.
The line is the one the source-off check replays, built by
:func:`plane_wave_line`; the sum is formed in numpy float64 from its float32
record on the host, before the run. No gradient flows into I_c. A bin where
|I_c(f)| is below ``RHO_LIMIT`` of its bound dt sum_n |i_c(n)| is refused.

Only transverse E channels (ey, ez) are divided, at normal incidence along x.
I_c is the incident E along e_hat for both: an ey and an ez channel on the
same x plane share one divisor. With ``polarization="s"`` or ``"p"`` at
``phi_deg != 0`` (theta 0), e_hat = (0, -sin phi, cos phi), so in an empty
box the quotient of an ez channel is cos phi and that of an ey channel
-sin phi, not 1.
"""
import numpy as np

RHO_LIMIT = 1.0e-4


def plane_wave_line(plan):
    """The closed-box line ``(cfg, state)`` for ``plan.n_steps``, as the runner builds it.

    ``line_steps`` is the record length on ``run`` and None on ``forward``;
    both bound the line at that length. Built under
    ``jax.ensure_compile_time_eval()`` from the declaration and the grid.
    """
    import jax

    from rfx.sources.tfsf import tfsf_from_declaration

    sim = plan.sim
    n = int(plan.n_steps)
    with jax.ensure_compile_time_eval():
        return tfsf_from_declaration(sim._tfsf, plan.grid, sim._freq_max, n_steps=n,
                                     line_steps=n if plan.caller == "run" else None)


def incident_divisors(records, bins, dt):
    """``(I, bound)``: I_c(f) for every record column, (nf, P) complex128, and dt sum |i_c(n)|, (P,).

    The float64 record DFT of :func:`rfx.measurement.ringdown_tail.plain_dft`
    times the E channel's unit stamp (:func:`rfx.measurement.dft.host_stamp`).
    """
    from rfx.measurement.dft import host_stamp
    from rfx.measurement.ringdown_tail import plain_dft

    rows = np.stack([np.asarray(r, dtype=np.float64) for r in records], axis=1)
    stamp = host_stamp(bins, dt, ["E"] * rows.shape[1])
    return plain_dft(rows, dt, bins) * stamp, dt * np.sum(np.abs(rows), axis=0)


def incident_normalization(plan):
    """The divisor (nf, P) complex128 of ``plan``'s identification channels, or None.

    Refuses, before any replay, a channel that is not ey or ez or that lies
    outside the total-field box; then replays one record per channel plane
    and refuses a bin where I_c is below ``RHO_LIMIT`` of its bound.
    """
    if plan.spec.probe_normalization is None:
        return None
    import jax

    from rfx.ringdown._admission import refuse_incident_bins, refuse_incident_channels
    from rfx.sources.tfsf import tfsf_injection_planes
    from rfx.sources.tfsf_discrete_plane_wave import normal_incident_record

    cfg, state = plane_wave_line(plan)
    keys = plan.identification_keys
    refuse_incident_channels(plan.caller, keys, tfsf_injection_planes(cfg))
    n = int(plan.n_steps)
    with jax.ensure_compile_time_eval():
        records = [normal_incident_record(cfg, state, n, plan.grid.dt, int(index[0]))
                   for _component, index in keys]
    return refuse_incident_bins(plan.caller, keys, records, plan.bins, plan.grid.dt)


def normalized(plan, spectra, dtype=None):
    """``spectra`` divided by ``plan``'s divisor, or unchanged without one.

    With ``dtype`` (a traced completion's working complex dtype) the divisor
    is cast to it first; otherwise the division is complex128 on the host.
    """
    divisor = getattr(plan, "incident_spectra", None)
    if divisor is None:
        return spectra
    if dtype is not None:
        import jax.numpy as jnp
        divisor = jnp.asarray(divisor, dtype=dtype)
    return spectra / divisor
