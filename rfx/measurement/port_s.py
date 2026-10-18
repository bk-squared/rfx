"""Port S extraction from one production record on either mesh."""

import jax.numpy as jnp


def port_s_from_record(channels, metas, *, wire_mode):
    """Return (receive, drive, frequency); unmeasured drive columns are NaN.

    Uniform metadata are port specs; graded metadata use _build_wp_meta's
    slots. The graded pre-injection reference is a record-only channel;
    its whole-port wire decomposition continues to receive vr=None.
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


def assemble_graded_ports_result(wire_sparams, wp_meta, records, request):
    """Package the same in-loop port spectra and named records on graded lanes."""
    import jax
    import numpy as np
    from rfx.measurement.time_records import from_owners
    from rfx.sparams._record_selection import SReadPorts
    context = request.context
    result = {}
    # ---- Extract full S-matrix column from wire port DFTs ----
    #
    # Each port has a V/I DFT pair. The wave decomposition below is
    # DIRECTION-FREE, and that is a physics statement, not a shortcut
    # (issue #673). The port's outward normal is not a degree of freedom
    # for a lumped gap V/I pair:
    #
    #   V = -E_comp * d_parallel      (the E edge's own length)
    #   I = (curl H)_comp * A_perp    (right-hand loop about the SAME axis)
    #
    # Neither expression contains the port's position or an outward
    # normal, so their RELATIVE sign is fixed by the sampling code and no
    # port placement can flip it. The E update at that edge enforces
    # eps*dE/dt + sigma*E = (curl H)_comp, so in phasor form, with
    # C = eps*A/d and the runner's sigma = n_live*d/(Z0*dp1*dp2),
    #
    #       V / I = -1 / (n_live/Z0 + j*w*C + Y_ext)
    #
    # for any passive Y_ext attached across the gap (Re Y_ext >= 0), at a
    # port that is NOT itself driven. Hence Z_in = -V/I sits in the right
    # half plane and
    #
    #       S = (Z_in - Z0)/(Z_in + Z0) = (V + Z0*I)/(V - Z0*I),
    #
    # which is the same ratio as a = (-V + Z0*I)/2, b = (-V - Z0*I)/2.
    # The reciprocal form maps that passive disk to its EXTERIOR, i.e. it
    # returns |S| >= 1 at EVERY bin on ANY passive structure — which is
    # what the pre-#673 direction switch did on "-x"/"-y". That is what
    # fixes the CONVENTION; the convention is then applied unconditionally,
    # exactly as the uniform lane does.
    # Scope note, measured: the |S| <= 1 consequence is a statement about an
    # UNDRIVEN port, and it is a property of the port CELL rather than of the
    # structure across the gap (vacuum, PEC plates shorting the gap and an
    # eps_r=10 slab filling it all read S11 = -0.600000 at n_live = 4).
    # That is exactly why the all-passive fallback below is INTENTIONALLY
    # frozen on this per-cell convention: no physical falsifier can gate a
    # load-independent reading, so it is a diagnostic/self-consistency
    # channel, not S_jj (S_jj requires driving port j) — and the
    # excite=False locks (lane-parity closed forms, sigma ORACLE 1/2) keep
    # their n_live sensitivity as witnesses. Do not "unify" it with the
    # driven diagonal below.
    #
    # DRIVEN diagonal (issue #764). At a genuinely excited port the per-cell
    # convention above stops tracking the load entirely — PROVENANCE, all
    # measured before the #764 fix: one geometry with only `excite` flipped
    # read S11(0.2 GHz) = -0.600000 passive against +0.999670-0.022145j
    # driven though the quasi-static input impedance is identical; the
    # 2-port MSL stub in tests/unit/sparams/test_twoport_wire_port.py reached
    # max|S11| = 4.648 (the reciprocal class that (-V - Z0*I)/(-V + Z0*I)
    # applied to a driven port yields); a matched 50 ohm load read
    # S11 = +0.35426 and a PEC short +0.26780. Root cause was the #313/#318
    # frame mismatch: V and I sampled at ONE cell measure the per-cell
    # Z0/n_live while the reflection formula references the whole-port Z0.
    # The fix is frame-consistent terminal quantities:
    #
    #   V_port = sum_live(-E_c * d_par,c)   (whole-gap line integral,
    #                                        PRIMAL metric per cell, #672)
    #   I      = Ampere loop at the LIVE-run midpoint cell (DUAL metrics)
    #   S_kk   = (V_port - Z0*I) / (V_port + Z0*I)
    #
    # where Z0 is the whole-port port.impedance — the impedance #318
    # physically realizes as the series termination (n_live cells of
    # sigma-folded Z0/n_live in series). Sense: from the pinned per-cell law
    # with equal per-cell injection I0, V_port + Z0*I = Z0*I0 is a
    # drive-only constant (the incident wave), and closing the loop through
    # the external DUT gives V_port = +Z_L*I and I = Z0*I0/(Z0+Z_L) — the
    # measured #683 circuit law. The opposite branch sign (V_port = -Z_L*I)
    # would put I = Z0*I0/(Z0-Z_L), divergent at a matched load — refuted.
    # Only the SUM is KVL/Faraday-constrained on the staggered grid; a
    # single cell is not (a PEC short forces sum V_c = 0 while V_mid stays
    # finite — why the short used to read +0.268: a structural error, not a
    # scale factor).
    # Quasi-static check on the PASSIVE path: with Y_ext -> 0 and
    # w*C*Z0/n_live << 1 the per-cell convention gives
    # S11 -> (1 - n_live)/(1 + n_live), measured to 5 decimals on both
    # lanes (tests/unit/nonuniform/test_nu_wire_port_lane_parity.py).
    #
    # `direction` is still carried on the port spec — the reference-plane
    # path (add_port(reference_plane_cells=...)) needs it for the outboard
    # sign — but it must NOT enter this wave split.
    #
    # Given these, with ONE excited port `k` the full k-th column of
    # the S-matrix is
    #       S[j, k] = b_j / a_k          (j = every port)
    # which reduces to the familiar S11 = b_k/a_k for j==k (reflection)
    # and to S21 = b_2/a_1 for j != k (transmission). Other columns of
    # the S matrix stay NaN — callers need to run additional sims
    # with different excited ports to fill them (reciprocity lets us
    # infer S12 from S21 for passive networks).
    #
    # Which (a, b) pair fills a GENUINELY excited column is the issue
    # #770 whole-port pair (see the in-loop derivation below); the
    # per-cell `_ab` split above survives only in the all-passive
    # diagnostic fallback.
    if context.use_wire_ports and wire_sparams is not None:
        result['s_params'] = port_s_from_record(wire_sparams, wp_meta, wire_mode=True)
        freqs = context.sp_freqs
        result['s_param_freqs'] = freqs if isinstance(freqs, jax.core.Tracer) else np.array(freqs)
        result['wire_sparams_raw'] = wire_sparams
        result['wire_sparams_meta'] = tuple(wp_meta)
    result['sparam_time_records'] = from_owners(records, request.port_owners)
    result['sparam_read_ids'] = SReadPorts(tuple(
        request.port_owners[i].id for i in request.port_acc_indices)
        if result.get('s_params') is not None else ())
    return result
