"""Admission for ring-down completion requests."""

from rfx.errors import RfxRefusal, RfxUnsupported
from rfx.ringdown import RingdownSpec


def refuse_run_request(sim, spec, *, devices, until_decay, compute_s_params,
                       caller: str = "run") -> None:
    """Raise, with the reason, for every run ``ringdown=`` does not cover.

    ``caller`` names the entry point in the messages (``"run"`` or
    ``"forward"``); the checks are the same.
    """
    if not isinstance(spec, RingdownSpec):
        raise TypeError(
            f"ringdown= takes a rfx.ringdown.RingdownSpec, got {type(spec).__name__}")
    if devices is not None and len(devices) > 1:
        raise RfxUnsupported(
            f"{caller}(ringdown=...) is not supported on the distributed lane "
            "(devices=): the port-channel probes and their check against the "
            "port accumulators are single-device only.")
    if until_decay is not None:
        raise RfxUnsupported(
            f"{caller}(ringdown=...) completes a record of fixed length; until_decay= "
            "stops the record on field energy instead, and the two are not "
            "combined. Pass n_steps= or num_periods=.")
    if compute_s_params is False:
        raise RfxRefusal(
            f"{caller}(ringdown=...) completes the port S-parameters, but "
            "compute_s_params=False asks for none.")
    if getattr(sim, "_mode", "3d") != "3d":
        raise RfxUnsupported(
            f"{caller}(ringdown=...) needs mode='3d'; this model is mode={sim._mode!r}.")
    from rfx import ringdown as _rd
    _rd._refuse_stencil_order(sim, caller)
    if getattr(sim, "_refinement", None) is not None:
        raise RfxUnsupported(
            f"{caller}(ringdown=...) is not supported on the subgridded lane.")
    if getattr(sim, "_periodic_axes", ""):
        raise RfxUnsupported(
            f"{caller}(ringdown=...) is not supported with periodic axes: rfx refuses "
            "wire-port S-parameters there.")
    if getattr(sim, "_tfsf", None) is not None:
        wf = getattr(sim._tfsf, "waveform", "")
        raise RfxUnsupported(
            f"{caller}(ringdown=...) is not supported with a TFSF plane wave"
            + (" driven by a continuous wave, which never turns off"
               if wf == "continuous_wave" else
               ": the total-field boundary keeps injecting while the auxiliary "
               "line still carries the pulse, so the injection waveform alone "
               "does not say when the source is off") + ".")
    if any(float(getattr(m, "chi3", 0.0) or 0.0) != 0.0
           for m in getattr(sim, "_materials", {}).values()):
        raise RfxUnsupported(
            f"{caller}(ringdown=...) is not supported with a Kerr (chi3) material: "
            "after the pulse the update is not linear, so the record is not a "
            "sum of fixed damped oscillations.")
    for attr, what in (("_msl_ports", "microstrip (add_msl_port)"),
                       ("_waveguide_ports", "waveguide (add_waveguide_port)"),
                       ("_coaxial_ports", "coaxial (add_coaxial_port)"),
                       ("_floquet_ports", "Floquet (add_floquet_port)")):
        if getattr(sim, attr, None):
            raise RfxUnsupported(
                f"{caller}(ringdown=...) completes wire-port S-parameters only; this "
                f"model has a {what} port.")
    wire = []
    for pe in sim._ports:
        if pe.is_soft_source:
            continue            # a soft source: checked for being off, below
        if pe.extent is None:
            raise RfxUnsupported(
                f"{caller}(ringdown=...) completes wire-port S-parameters only; the "
                f"port at {tuple(pe.position)} is a one-cell lumped port "
                "(add_port without extent=).")
        if getattr(pe, "reference_plane_cells", None) is not None:
            raise RfxUnsupported(
                f"{caller}(ringdown=...) does not cover the reference-plane waves of "
                f"the wire port at {tuple(pe.position)} (reference_plane_cells=).")
        wire.append(pe)
    if not wire:
        raise RfxRefusal(
            f"{caller}(ringdown=...) needs a wire port (add_port(..., extent=...), "
            "impedance > 0); this model has none.")
    if not any(bool(pe.excite) for pe in wire):
        raise RfxRefusal(
            f"{caller}(ringdown=...) needs a driven wire port: with every port passive "
            "run() reports a per-cell diagnostic, not a scattering parameter.")



def refuse_run_lane(lane: str, n_wire_ports: int) -> None:
    """Lane-dependent refusals, once ``run()`` has chosen its lane."""
    if lane not in ("run_uniform", "run_nonuniform"):
        raise RfxUnsupported(
            f"run(ringdown=...) supports the uniform and graded-mesh lanes; this "
            f"run takes the {lane!r} lane.")
    if lane == "run_uniform" and n_wire_ports > 1:
        raise RfxUnsupported(
            "run(ringdown=...) on the uniform lane supports one wire port: with "
            "several, run() assembles the S-matrix from separate per-port drive "
            "runs that the port-channel probes do not see. Use a graded mesh "
            "(any dx/dy/dz profile), where one run fills the driven column.")



def refuse_forward_request(sim, spec, *, distributed, emit_time_series) -> None:
    """Raise for every ``forward(ringdown=...)`` not covered: :func:`refuse_run_request`'s checks,
    ``distributed=True``, ``emit_time_series=False``, and a backend without host callbacks."""
    from rfx._traced_check import backend_class, require_host_callbacks
    if not (backend := backend_class()).host_callbacks:
        raise RfxUnsupported(
            f"forward(ringdown=): pole identification: this backend ({backend.name}) cannot run host code inside compiled code")
    require_host_callbacks("forward(ringdown=)", "pole identification")
    if not isinstance(spec, RingdownSpec):
        raise TypeError(
            f"ringdown= takes a rfx.ringdown.RingdownSpec, got {type(spec).__name__}")
    if distributed:
        raise RfxUnsupported(
            "forward(ringdown=...) is not supported with distributed=True: the "
            "port-channel probes and their check against the port accumulators "
            "are single-device only.")
    refuse_run_request(sim, spec, devices=None, until_decay=None,
                       compute_s_params=True, caller="forward")
    if not emit_time_series:
        raise RfxRefusal(
            "forward(ringdown=...) reads the port voltage and current from probe "
            "series on the port's own cells; emit_time_series=False records none.")



def refuse_forward_lane(lane: str, n_wire_ports: int, *, port_s11_freqs) -> None:
    """Lane-dependent refusals, once ``forward()`` has chosen its lane."""
    if lane not in ("fwd_uniform", "fwd_nonuniform"):
        raise RfxUnsupported(
            f"forward(ringdown=...) supports the uniform and graded-mesh lanes; "
            f"this call takes the {lane!r} lane.")
    if lane == "fwd_uniform":
        if n_wire_ports > 1:
            raise RfxUnsupported(
                "forward(ringdown=...) on the uniform lane supports one wire port, "
                "as run(ringdown=...) does. Use a graded mesh (any dx/dy/dz "
                "profile), where one run fills the driven column.")
    if port_s11_freqs is None:
        raise RfxRefusal(
            "forward(ringdown=...) needs port_s11_freqs=: the completed "
            "S-parameters are evaluated at those bins, like forward()'s own "
            "s_params, which forward() computes only on request.")



def refuse_until_identified(sim, spec, until_identified, *, until_decay,
                            snapshot) -> bool:
    """Raise, with the reason, for every ``run(until_identified=...)`` not covered.

    Returns the flag. The rest of what ``until_identified=True`` refuses is
    what ``ringdown=`` refuses (:func:`refuse_run_request`,
    :func:`refuse_run_lane`) and, before the run, what ``run(n_steps=n_steps,
    ringdown=spec)`` would refuse for the maximum record (:class:`RingdownStop`).
    """
    if not isinstance(until_identified, bool):
        raise TypeError(
            f"until_identified must be True or False, got {until_identified!r}")
    if not until_identified:
        return False
    if spec is None:
        raise RfxRefusal(
            "run(until_identified=True) ends the run once the ring-down "
            "completion of the wire-port S-parameters is witnessed; it needs "
            "ringdown=RingdownSpec(...).")
    if until_decay is not None:
        raise RfxRefusal(
            "run(until_identified=True) and until_decay= are two rules for "
            "where the record ends (the completion's witnesses, the field "
            "energy); pass one. With until_identified, n_steps= is the longest "
            "record the run may take.")
    if snapshot is not None:
        raise RfxUnsupported(
            "run(until_identified=True) is not supported with snapshot=: the "
            "frame axes are laid out from n_steps before the run, and the early "
            "stop ends the record at a length chosen during it.")
    for attr, what in (("_dft_planes", "a DFT plane (add_dft_plane_probe)"),
                       ("_flux_monitors", "a flux monitor (add_flux_monitor)")):
        if getattr(sim, attr, None):
            raise RfxUnsupported(
                f"run(until_identified=True) is not supported with {what}: its "
                "accumulator is built for the record length n_steps before the "
                "run (the DFT window and its step count), and the early stop ends "
                "the record at a length chosen during it. Use run(n_steps=..., "
                "ringdown=...) with a fixed record.")
    for attr, what in (("_ntff", "an NTFF box (add_ntff_box)"),
                       ("_current_moments",
                        "a current-moment monitor (add_current_moment_monitor)")):
        if getattr(sim, attr, None):
            raise RfxUnsupported(
                f"run(until_identified=True) is not supported with {what}: the "
                "early stop chooses the record from the wire-port S-parameters "
                "alone and completes only those, so the far field would come "
                "from a record cut while the structure still rings. Use "
                "run(n_steps=..., ringdown=...) with a record long enough for "
                "the far field.")
    return True

