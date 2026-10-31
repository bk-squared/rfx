from __future__ import annotations

import jax.numpy as jnp
from rfx.probes.probes import _ampere_loop


def accumulate_lumped_sparam_dft(*, carry, dt, dx_f, ex_f_new, ey_f_new, ez_f_new, lumped_sparam_freqs_arr, lumped_sparam_meta, st_f, step_idx):
    phase = jnp.exp(
        -1j
        * 2.0
        * jnp.pi
        * lumped_sparam_freqs_arr
        * (step_idx.astype(jnp.float32) * dt)
    )
    # NOTE: the Yee half-step current phase correction
    # `rfx.core.dft_utils.half_step_current_phase` is NOT applied on
    # this LUMPED lane, and this lane alone keeps the pre-decision
    # sampling and diagonal conventions. Its derivation needs
    # E = E^{n+1} at the sample. The uniform lumped lane now has
    # that (post-injection, decided 2026-09-21 by
    # scripts/diagnostics/lumped_port_known_load_line.py, which also
    # moved the diagonal to the driven terminal reflection). On THIS
    # runner the slot is not fixed — whether this block runs pre- or
    # post-injection depends on the runtime
    # ``inject_sources_before_e_coupling`` flag (see the fine-grid
    # source injection below) — so no single dt/2 offset is
    # derivable here at all, and the same argument blocks the slot
    # move. rfx/runners/subgridded.py warns the caller that this
    # diagnostic S-matrix is on the old convention.
    v_dft = carry["lumped_sparam_v"]
    i_dft = carry["lumped_sparam_i"]

    def _sample_lumped_vi(i, j, k, comp):
        # #692: the SHARED Ampere loop. This used to be a fourth
        # verbatim copy of the six branches with a raw `h[i-1]`, so a
        # lumped port at index 0 on a back-read axis integrated H
        # from the OPPOSITE face of the fine domain — a cell the
        # local loop does not enclose.
        #
        # `periodic` is hard-coded False on all three axes because
        # this runner has no periodic lane: `periodic` does not
        # appear anywhere in this module, and the subgrid E/H updates
        # it drives are built without one. The other guard
        # `_bwd_h` keeps — a length-1 (2-D) axis — is driven off
        # `h.shape[axis]`, not off these flags, so it still applies.
        _e_new = {"ex": ex_f_new, "ey": ey_f_new, "ez": ez_f_new}
        if comp not in _e_new:
            raise ValueError(f"Unknown lumped S-parameter component: {comp!r}")
        voltage = -_e_new[comp][i, j, k] * dx_f
        current = _ampere_loop(
            st_f, (i, j, k), comp, dx_f, (False, False, False))
        return voltage, current

    for idx_lp, (li, lj, lk, lc, _z0) in enumerate(lumped_sparam_meta):
        voltage, current = _sample_lumped_vi(li, lj, lk, lc)
        v_dft = v_dft.at[idx_lp].add(voltage * phase * dt)
        i_dft = i_dft.at[idx_lp].add(current * phase * dt)
    return v_dft, i_dft
