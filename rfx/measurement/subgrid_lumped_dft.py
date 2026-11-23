from __future__ import annotations

import jax.numpy as jnp
from rfx.probes.probes import _ampere_loop


def accumulate_lumped_sparam_dft(*, carry, dt, dx_f, ex_f_new, ey_f_new, ez_f_new, lumped_sparam_freqs_arr, lumped_sparam_meta, st_f, step_idx):
    """The electric-field sample can precede or follow source injection according to the runner's ``inject_sources_before_e_coupling`` flag."""
    phase = jnp.exp(
        -1j
        * 2.0
        * jnp.pi
        * lumped_sparam_freqs_arr
        * (step_idx.astype(jnp.float32) * dt)
    )
    v_dft = carry["lumped_sparam_v"]
    i_dft = carry["lumped_sparam_i"]

    def _sample_lumped_vi(i, j, k, comp):
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
