"""Port S extraction from one production record on either mesh."""

import jax.numpy as jnp


def port_s_from_record(channels, metas, *, wire_mode):
    """Return (receive, drive, frequency); unmeasured drive columns are NaN.

    Uniform metadata are port specs; graded metadata use _build_wp_meta's
    slots. All-lumped records share the uniform lumped decomposition;
    graded records containing wires continue to receive vr=None.
    """
    from rfx.probes.probes import (
        decompose_lumped_s_matrix, decompose_wire_s_matrix, driven_port_reflection,
    )

    channels = tuple(channels)
    specs = tuple(metas)
    graded = not hasattr(specs[0], "impedance")
    metas = specs if graded else tuple(
        (None, None, None, None, spec.impedance, None, None, spec.excite)
        for spec in specs)
    n_wp = len(channels)
    nf = channels[0][0].shape[-1]
    genuinely_excited = [idx for idx, meta in enumerate(metas) if meta[7]]
    if len(genuinely_excited) > 1:
        raise NotImplementedError(
            "Port S extraction requires at most one driven port per record (#1660).")
    if wire_mode and not graded and n_wp == 1 and genuinely_excited:
        # run()'s one-wire main-record path uses this unnormalized wave pair.
        return driven_port_reflection(
            channels[0][3], channels[0][1], specs[0].impedance)[None, None, :]
    from rfx.measurement.port_s_contract import require_driven_record_s
    require_driven_record_s(meta[7] for meta in metas)

    dtype = jnp.result_type(*(ch[0] for ch in channels), jnp.complex64)
    v = jnp.zeros((n_wp, n_wp, nf), dtype=dtype)
    i = jnp.zeros_like(v)
    vp = jnp.zeros_like(v)
    if graded and not wire_mode and any(len(ch) < 5 for ch in channels):
        raise ValueError("Graded lumped-port S requires a pre-injection reference channel.")
    vr = None if graded and wire_mode else jnp.zeros_like(v)
    d = genuinely_excited[0]
    for receive, ch in enumerate(channels):
        v = v.at[d, receive].set(ch[0])
        i = i.at[d, receive].set(ch[1])
        vp = vp.at[d, receive].set(ch[3] if wire_mode else ch[0])
        if vr is not None:
            vr = vr.at[d, receive].set(ch[4] if graded or wire_mode else ch[2])
    z0 = jnp.asarray([meta[4] for meta in metas])
    if wire_mode:
        counts = [len(meta[13]) for meta in specs] if graded else [
            len(spec.live_cells or ()) or 1 for spec in specs]
        S = decompose_wire_s_matrix(v, i, z0, counts, v_port=vp, v_ref=vr)
    else:
        S = decompose_lumped_s_matrix(v, i, z0, v_ref=vr)
    driven = jnp.asarray([meta[7] for meta in metas])
    missing = jnp.asarray(complex(float("nan"), float("nan")), dtype=S.dtype)
    return jnp.where(driven[None, :, None], S, missing)


def observed_port_s(s_params, pairs):
    """Select measured columns for health checks on one-record S outputs."""
    indices = [k for k, (meta, _) in enumerate(pairs)
               if (meta.excite if hasattr(meta, "impedance") else meta[7])]
    return s_params[:, indices, :] if indices else s_params


def assemble_graded_ports_result(wire_sparams, wp_meta, records, request):
    """Package the same in-loop port spectra and named records on graded lanes."""
    import jax
    import numpy as np
    from rfx.measurement.time_records import from_owners
    from rfx.sparams._record_selection import SReadPorts
    context = request.context
    result = {}
    # Raw port spectra are termination diagnostics even without a driven column.
    # Admission resolves default requests to S=None before stepping; the low-level
    # extractor itself refuses an all-passive record if called directly.
    if context.use_wire_ports and wire_sparams is not None:
        freqs = context.sp_freqs
        result['s_param_freqs'] = freqs if isinstance(freqs, jax.core.Tracer) else np.array(freqs)
        result['wire_sparams_raw'] = wire_sparams
        result['wire_sparams_meta'] = tuple(wp_meta)
        from rfx.measurement.port_records import pack_port_records
        # Ringdown's synthetic completion has only wire metadata and no owners.
        kinds = (tuple(request.port_owners[i].kind for i in request.port_acc_indices)
                 if request.port_acc_indices else tuple(p['kind'] for p in context.wire_ports))
        # A run() whose S comes from the drive scan keeps the records, not record S.
        result['s_params'] = (port_s_from_record(
            wire_sparams, wp_meta, wire_mode=any(kind == 'wire' for kind in kinds))
            if any(meta[7] for meta in wp_meta)
            and all(p.get('record_s', True) for p in getattr(context, 'wire_ports', ())) else None)
        result['wire_port_sparams'], result['lumped_port_sparams'] = pack_port_records(
            kinds, wire_sparams, wp_meta, graded=True, freqs=context.sp_freqs)
    result['sparam_time_records'] = from_owners(records, request.port_owners)
    result['sparam_read_ids'] = SReadPorts(tuple(
        request.port_owners[i].id for i in request.port_acc_indices)
        if result.get('s_params') is not None else ())
    return result
