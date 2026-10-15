"""Result assembly for graded-grid solves and ringdown records."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp

from rfx.measurement.port_registration import graded_s_voltage
from rfx.measurement.time_records import with_graded_records

if TYPE_CHECKING:
    from rfx.nonuniform import _NUScanSetup


def assemble_graded_result(setup: _NUScanSetup, final: dict, time_series, n_record=None) -> dict:
    """Assemble the NU result dict (pure code motion from run_nonuniform,
    #383). Shared by :func:`run_nonuniform` and
    :func:`run_nonuniform_until_decay` so the two paths return the exact
    same schema (state, time_series, dt, conditional dft_planes /
    flux_monitors / ntff_data / waveguide_ports / s_params...)."""
    dt = setup.dt
    dft_planes = setup.dft_planes
    flux_monitors = setup.flux_monitors
    wire_ports = setup.wire_ports
    waveguide_meta = setup.waveguide_meta
    wp_meta = setup.wp_meta
    sp_freqs = setup.sp_freqs
    use_wire_ports = setup.use_wire_ports
    use_dft_planes = setup.use_dft_planes
    use_flux_monitors = setup.use_flux_monitors
    use_lumped_rlc = setup.use_lumped_rlc
    use_ntff = setup.use_ntff
    use_waveguide_ports = setup.use_waveguide_ports

    result = {
        "dft_time_records": tuple(r[:n_record] for r in final.get("dft_time_records", ())),
        "state": final["fdtd"],
        "time_series": time_series,
        "dt": dt,
    }

    # Surface final RLC ADE states (per element).
    if use_lumped_rlc:
        result["rlc_states"] = tuple(final["rlc_states"])

    # Repack DFT plane probes with their final accumulators.
    if use_dft_planes:
        result["dft_planes"] = tuple(
            probe._replace(accumulator=acc)
            for probe, acc in zip(dft_planes, final["dft_planes"])
        )

    # Repack flux monitors with their final E/H DFT accumulators so the
    # caller can call flux_spectrum() on them (same schema as uniform).
    if use_flux_monitors:
        result["flux_monitors"] = tuple(
            mon._replace(e1_dft=e1, e2_dft=e2, h1_dft=h1, h2_dft=h2)
            for mon, (e1, e2, h1, h2) in zip(
                flux_monitors, final["flux_monitors"]
            )
        )

    # Surface final NTFF DFT accumulators
    if use_ntff:
        result["ntff_data"] = final["ntff"]

    # Surface the in-loop block current moments and the slab they belong to.
    # getattr: run(ringdown=) assembles S through here with a stand-in setup
    # (rfx.ringdown._assemble) that carries no monitor.
    if getattr(setup, "use_current_moments", False):
        result["current_moment_data"] = final["current_moments"]
        result["current_moment_monitor"] = setup.current_moments

    # Surface final waveguide-port configs (with recorded modal V/I
    # time series; spectra are extracted post-scan via rect-DFT).
    if use_waveguide_ports:
        result["waveguide_ports"] = tuple(
            cfg_meta._replace(
                v_probe_t=accs[0],
                v_ref_t=accs[1],
                i_probe_t=accs[2],
                i_ref_t=accs[3],
                v_inc_t=accs[4],
                n_steps_recorded=accs[5],
            )
            for cfg_meta, accs in zip(
                waveguide_meta, final["waveguide_port_accs"]
            )
        )

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
    # the S matrix stay zero — callers need to run additional sims
    # with different excited ports to fill them (reciprocity lets us
    # infer S12 from S21 for passive networks).
    #
    # Which (a, b) pair fills a GENUINELY excited column is the issue
    # #770 whole-port pair (see the in-loop derivation below); the
    # per-cell `_ab` split above survives only in the all-passive
    # diagnostic fallback.
    if use_wire_ports and "wire_sparams" in final:
        import numpy as _np
        n_wp = len(wire_ports)
        nf = len(sp_freqs)
        # Tracer-safe accumulation (issue #70): build S with jnp + .at[].set
        # so jax.grad on an objective that pulls through the wire_sparams
        # extractor does not hit TracerArrayConversionError. Concrete-path
        # callers still get a jnp.ndarray that numpy consumers accept via
        # np.asarray(...) (jnp arrays implement __array__).
        S = jnp.zeros((n_wp, n_wp, nf), dtype=jnp.complex64)

        # Pick the excited ports as the filled columns. If no port is
        # excited (all passive), fall back to the legacy diagnostic
        # extraction (no meaningful S-matrix in that case).
        # ``genuinely_excited`` (issues #764 + #770) keys the whole-port
        # wave pair to meta[7] ONLY — the all-passive fallback stays on
        # the frozen per-cell convention (see the scope note above).
        genuinely_excited = [
            idx for idx, meta in enumerate(wp_meta) if meta[7]]
        excited_idx = genuinely_excited
        if not excited_idx:
            excited_idx = list(range(n_wp))   # legacy: treat all as self-excited

        def _ab(v_dft, i_dft, z0):
            """Return (a_incoming, b_outgoing) at one port.

            Direction-free, written in the uniform lane's sign so it reads
            line-for-line against rfx/runners/uniform.py (the overall -1
            cancels in the ratio b/a). See the derivation above (#673).
            """
            zi = z0 * i_dft
            return (-v_dft + zi) / 2.0, (-v_dft - zi) / 2.0

        ab_per_port = []
        for (v_dft, i_dft, _, _vp), meta in zip(final["wire_sparams"],
                                                wp_meta):
            z0 = meta[4]
            a, b = _ab(v_dft, i_dft, z0)
            ab_per_port.append((a, b))

        # All-passive fallback ONLY: the frozen per-cell diagnostic
        # convention (see the scope note above — no physical falsifier
        # can gate a load-independent reading; do not "unify").
        if not genuinely_excited:
            for k in excited_idx:
                a_k = ab_per_port[k][0]
                safe_a_k = jnp.where(jnp.abs(a_k) > 0, a_k,
                                     jnp.ones_like(a_k))
                for j in range(n_wp):
                    b_j = ab_per_port[j][1]
                    S = S.at[j, k, :].set(b_j / safe_a_k)

        # Issue #764 + #770: every GENUINELY excited column is the
        # whole-port wave pair, frame-consistent across diagonal and
        # off-diagonal (adjudicated against external physics 2026-08-29,
        # docs/design_notes/issue770_offdiag_adjudication_predeclaration.md
        # + results note):
        #
        #   a_k = (V_port,k + Z0*I_k) / (2*sqrt(Z0))   # measured drive-only
        #                                              # constant Z0*I0 (#683)
        #   b_j = (V_port,j - Z0*I_j) / (2*sqrt(Z0))   # outgoing at port j
        #   S[j, k] = b_j / a_k    (diagonal reduces to the #764
        #                           (V_port - Z0*I)/(V_port + Z0*I))
        #
        # The old mixed split (per-cell midpoint v against the whole-port
        # Z0, the open defect #764 section 6 named) measured neither
        # frame on the canonical thru (|S21| 0.65-0.97, drifting); the
        # whole-port pair measured |S21| = 0.934-0.995 vs the external
        # flux referee 0.971-0.997, power closure deficit 0.9-4.4% vs
        # the 0.2-4.0% flux gap, reciprocity 2.67e-4, and lane parity
        # 3.97e-6 against the uniform decomposer (#770 harness).
        # Global receive sign: +1, DC-witness pinned (S21(DC) -> +1,
        # dev -0.058/-0.115 rad at 0.5/1 GHz; flipped channel at ~pi).
        for k in genuinely_excited:
            v_port_k = graded_s_voltage(final["wire_sparams"], wire_ports, k)
            i_k = final["wire_sparams"][k][1]
            z0_k = wp_meta[k][4]
            a_k = v_port_k + z0_k * i_k
            safe_a_k = jnp.where(jnp.abs(a_k) > 0, a_k, jnp.ones_like(a_k))
            sqrt_z0_k = jnp.sqrt(z0_k)
            for j in range(n_wp):
                v_port_j = graded_s_voltage(final["wire_sparams"], wire_ports, j)
                i_j = final["wire_sparams"][j][1]
                z0_j = wp_meta[j][4]
                b_j = (v_port_j - z0_j * i_j) * (
                    sqrt_z0_k / jnp.sqrt(z0_j))
                S = S.at[j, k, :].set(b_j / safe_a_k)

        result["s_params"] = S
        result["s_param_freqs"] = _np.array(sp_freqs)
        # Raw per-port DFT accumulators (v_dft, i_dft, v_inc_dft,
        # v_port_dft), surfaced for diagnostics/validation harnesses
        # (issue #764; mirrors the uniform lane's raw-acc access via
        # forward()'s wire_port_sparams).
        result["wire_sparams_raw"] = final["wire_sparams"]
        # The static per-port metadata those accumulators were recorded
        # with, in the same order (:func:`_build_wp_meta` slots). The
        # accumulators alone say nothing about which port they came from
        # or at what impedance; ``run_nonuniform_path`` zips the two into
        # the ``(meta, accs)`` pairs the uniform lane's
        # ``Result.wire_port_sparams`` already carries. The existing
        # ``wire_sparams_raw`` shape is unchanged — validation harnesses
        # index it per port.
        result["wire_sparams_meta"] = tuple(wp_meta)

    return with_graded_records(result, setup, final, n_record)
