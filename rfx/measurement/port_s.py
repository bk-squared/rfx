"""Port S extraction from one production record on either mesh."""

import jax.numpy as jnp


def port_s_from_record(channels, metas, *, wire_mode):
    """Return (receive, drive, frequency); unmeasured drive columns are NaN.

    Uniform metadata are port specs; graded metadata use _build_wp_meta's
    slots. Graded wire records have no pre-injection reference channel; the
    whole-port wire decomposition does not consume that reference.
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
    if not genuinely_excited:
        S = jnp.zeros((n_wp, n_wp, nf), dtype=jnp.complex64)
        excited_idx = list(range(n_wp))
        # Normalize lumped tuples to the wire tuple consumed by the frozen block.
        if not wire_mode:
            channels = tuple((ch[0], ch[1], ch[2], ch[0]) for ch in channels)
        if wire_mode:
            channels = tuple(ch[:4] for ch in channels)

        def _ab(v_dft, i_dft, z0):
            """Return (a_incoming, b_outgoing) at one port.

            Direction-free, written in the uniform lane's sign so it reads
            line-for-line against rfx/runners/uniform.py (the overall -1
            cancels in the ratio b/a). See the derivation above (#673).
            """
            zi = z0 * i_dft
            return (-v_dft + zi) / 2.0, (-v_dft - zi) / 2.0

        ab_per_port = []
        for (v_dft, i_dft, _, _vp), meta in zip(channels,
                                                metas):
            z0 = meta[4]
            a, b = _ab(v_dft, i_dft, z0)
            ab_per_port.append((a, b))

        # All-passive fallback ONLY: the frozen per-cell diagnostic
        # convention (see the scope note above — no physical falsifier
        # can gate a load-independent reading; do not "unify").
        # It is the all-passive diagnostic; a record with a driven port uses run()'s decomposer.
        if not genuinely_excited:
            for k in excited_idx:
                a_k = ab_per_port[k][0]
                safe_a_k = jnp.where(jnp.abs(a_k) > 0, a_k,
                                     jnp.ones_like(a_k))
                for j in range(n_wp):
                    b_j = ab_per_port[j][1]
                    S = S.at[j, k, :].set(b_j / safe_a_k)

        return S

    dtype = jnp.result_type(*(ch[0] for ch in channels), jnp.complex64)
    v = jnp.zeros((n_wp, n_wp, nf), dtype=dtype)
    i = jnp.zeros_like(v)
    vp = jnp.zeros_like(v)
    vr = None if graded else jnp.zeros_like(v)
    d = genuinely_excited[0]
    for receive, ch in enumerate(channels):
        v = v.at[d, receive].set(ch[0])
        i = i.at[d, receive].set(ch[1])
        vp = vp.at[d, receive].set(ch[3] if wire_mode else ch[0])
        if vr is not None:
            vr = vr.at[d, receive].set(ch[4] if wire_mode else ch[2])
    z0 = jnp.asarray([meta[4] for meta in metas])
    if wire_mode:
        counts = [len(meta[13]) for meta in specs] if graded else [
            len(spec.live_cells or ()) or 1 for spec in specs]
        S = decompose_wire_s_matrix(v, i, z0, counts, v_port=vp, v_ref=vr)
    else:
        if vr is None:
            raise ValueError("Graded lumped-port S requires a pre-injection reference channel.")
        S = decompose_lumped_s_matrix(v, i, z0, v_ref=vr)
    driven = jnp.asarray([meta[7] for meta in metas])
    missing = jnp.asarray(complex(float("nan"), float("nan")), dtype=S.dtype)
    return jnp.where(driven[None, :, None], S, missing)


def observed_port_s(s_params, pairs):
    """Select measured columns for health checks on one-record S outputs."""
    indices = [k for k, (meta, _) in enumerate(pairs)
               if (meta.excite if hasattr(meta, "impedance") else meta[7])]
    return s_params[:, indices, :] if indices else s_params
