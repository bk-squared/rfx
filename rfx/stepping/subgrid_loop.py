from __future__ import annotations

import jax.numpy as jnp
from rfx.core.yee import FDTDState
from rfx.stepping import update_h, update_e
from rfx.boundaries.pec import apply_pec, apply_pec_mask
from rfx.boundaries.cpml import apply_cpml_h, apply_cpml_e
from rfx.subgridding.sbp_sat_3d import _shared_node_coupling_3d


def run_subgridded_loop(*, _log_interval, _t0, _time, cpml_axes, cpml_params, cpml_state, dt, dx_c, dx_f, ex_c, ex_f, ey_c, ey_f, ez_c, ez_f, grid_c, high_wall_c, high_wall_f, hx_c, hx_f, hy_c, hy_f, hz_c, hz_f, mats_c, mats_f, n_steps, pec_mask_c, pec_mask_f, probe_components, probe_indices_f, sources_f, subgrid_config, time_series):
    for step in range(n_steps):
        if step % _log_interval == 0 and step > 0:
            elapsed = _time.time() - _t0
            rate = step / elapsed
            eta = (n_steps - step) / rate
            max_ez = float(jnp.max(jnp.abs(ez_f)))
            print(f"  step {step}/{n_steps} ({step/n_steps*100:.0f}%) "
                  f"| {rate:.0f} steps/s | ETA {eta:.0f}s | max|Ez_f|={max_ez:.3e}")

        st_c = FDTDState(ex=ex_c, ey=ey_c, ez=ez_c,
                         hx=hx_c, hy=hy_c, hz=hz_c,
                         step=jnp.array(step, dtype=jnp.int32))
        st_c = update_h(st_c, mats_c, dt, dx_c, high_wall=high_wall_c)
        st_c, cpml_state = apply_cpml_h(st_c, cpml_params, cpml_state, grid_c, cpml_axes,
                                        materials=mats_c)

        st_f = FDTDState(ex=ex_f, ey=ey_f, ez=ez_f,
                         hx=hx_f, hy=hy_f, hz=hz_f,
                         step=jnp.array(step, dtype=jnp.int32))
        st_f = update_h(st_f, mats_f, dt, dx_f, high_wall=high_wall_f)

        st_c = update_e(st_c, mats_c, dt, dx_c, high_wall=high_wall_c)
        st_c, cpml_state = apply_cpml_e(st_c, cpml_params, cpml_state, grid_c, cpml_axes,
                                        materials=mats_c)
        if pec_mask_c is not None:
            st_c = apply_pec_mask(st_c, pec_mask_c)
        st_c = apply_pec(st_c)

        st_f = update_e(st_f, mats_f, dt, dx_f, high_wall=high_wall_f)
        if pec_mask_f is not None:
            st_f = apply_pec_mask(st_f, pec_mask_f)

        (ex_c, ey_c, ez_c), (ex_f, ey_f, ez_f) = _shared_node_coupling_3d(
            (st_c.ex, st_c.ey, st_c.ez),
            (st_f.ex, st_f.ey, st_f.ez),
            subgrid_config,
        )
        hx_c, hy_c, hz_c = st_c.hx, st_c.hy, st_c.hz
        hx_f, hy_f, hz_f = st_f.hx, st_f.hy, st_f.hz

        for src_i, src_j, src_k, src_comp, src_waveform in sources_f:
            if src_comp == "ez":
                ez_f = ez_f.at[src_i, src_j, src_k].add(float(src_waveform[step]))
            elif src_comp == "ex":
                ex_f = ex_f.at[src_i, src_j, src_k].add(float(src_waveform[step]))
            elif src_comp == "ey":
                ey_f = ey_f.at[src_i, src_j, src_k].add(float(src_waveform[step]))

        for p_idx, (pi, pj, pk) in enumerate(probe_indices_f):
            comp = probe_components[p_idx]
            if comp == "ez":
                time_series[step, p_idx] = float(ez_f[pi, pj, pk])
            elif comp == "ex":
                time_series[step, p_idx] = float(ex_f[pi, pj, pk])
            elif comp == "ey":
                time_series[step, p_idx] = float(ey_f[pi, pj, pk])
    return ex_c, ex_f, ey_c, ey_f, ez_c, ez_f, hx_c, hx_f, hy_c, hy_f, hz_c, hz_f
