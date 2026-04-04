"""Subgridded simulation runner.

Runs a coupled coarse+fine FDTD simulation using SBP-SAT subgridding.
CPML is applied on the coarse grid boundaries. Sources and probes
operate on the fine grid (where the structure of interest resides).
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from rfx.core.yee import (
    FDTDState, MaterialArrays, update_h, update_e,
)
from rfx.boundaries.pec import apply_pec, apply_pec_mask
from rfx.boundaries.cpml import init_cpml, apply_cpml_h, apply_cpml_e
from rfx.grid import Grid
from rfx.subgridding.sbp_sat_3d import (
    SubgridConfig3D, _shared_node_coupling_3d,
)


def run_subgridded(
    grid_c: Grid,
    mats_c: MaterialArrays,
    grid_f,  # unused, kept for API compat
    mats_f: MaterialArrays,
    subgrid_config: SubgridConfig3D,
    n_steps: int,
    *,
    pec_mask_c=None,
    pec_mask_f=None,
    sources_f: list | None = None,
    probe_indices_f: list | None = None,
    probe_components: list | None = None,
    cpml_axes: str = "xyz",
) -> dict:
    """Run a subgridded FDTD simulation.

    The coarse grid covers the full domain with CPML boundaries.
    The fine grid covers the refinement region (e.g., substrate).
    Sources and probes are on the fine grid.

    Parameters
    ----------
    grid_c : Grid — coarse grid (full domain)
    mats_c : MaterialArrays — coarse materials
    grid_f : Grid — fine grid (refinement region only, no CPML)
    mats_f : MaterialArrays — fine materials
    subgrid_config : SubgridConfig3D
    n_steps : int
    pec_mask_c, pec_mask_f : boolean arrays or None
    sources_f : list of (i, j, k, component, waveform_array)
    probe_indices_f : list of (i, j, k) on fine grid
    probe_components : list of component names
    cpml_axes : CPML axes for coarse grid

    Returns
    -------
    dict with keys: state_c, state_f, time_series, config
    """
    sources_f = sources_f or []
    probe_indices_f = probe_indices_f or []
    probe_components = probe_components or []

    dt = subgrid_config.dt
    dx_c = subgrid_config.dx_c
    dx_f = subgrid_config.dx_f

    # Override coarse grid dt to match the subgrid global timestep
    # (fine grid CFL is more restrictive than coarse grid CFL)
    grid_c.dt = dt

    # Initialize CPML on coarse grid
    cpml_params, cpml_state = init_cpml(grid_c)

    # Initialize field states
    shape_c = (subgrid_config.nx_c, subgrid_config.ny_c, subgrid_config.nz_c)
    shape_f = (subgrid_config.nx_f, subgrid_config.ny_f, subgrid_config.nz_f)

    z = lambda s: jnp.zeros(s, dtype=jnp.float32)
    # Coarse fields
    ex_c, ey_c, ez_c = z(shape_c), z(shape_c), z(shape_c)
    hx_c, hy_c, hz_c = z(shape_c), z(shape_c), z(shape_c)
    # Fine fields
    ex_f, ey_f, ez_f = z(shape_f), z(shape_f), z(shape_f)
    hx_f, hy_f, hz_f = z(shape_f), z(shape_f), z(shape_f)

    # Time series storage
    n_probes = len(probe_indices_f)
    time_series = np.zeros((n_steps, max(n_probes, 1)), dtype=np.float32)

    import time as _time
    _t0 = _time.time()
    _log_interval = max(n_steps // 20, 100)  # log ~20 times

    for step in range(n_steps):
        if step % _log_interval == 0 and step > 0:
            elapsed = _time.time() - _t0
            rate = step / elapsed
            eta = (n_steps - step) / rate
            max_ez = float(jnp.max(jnp.abs(ez_f)))
            print(f"  step {step}/{n_steps} ({step/n_steps*100:.0f}%) "
                  f"| {rate:.0f} steps/s | ETA {eta:.0f}s | max|Ez_f|={max_ez:.3e}")

        # === Coarse grid: H update ===
        st_c = FDTDState(ex=ex_c, ey=ey_c, ez=ez_c,
                         hx=hx_c, hy=hy_c, hz=hz_c,
                         step=jnp.array(step, dtype=jnp.int32))
        st_c = update_h(st_c, mats_c, dt, dx_c)
        st_c, cpml_state = apply_cpml_h(st_c, cpml_params, cpml_state, grid_c, cpml_axes)

        # === Fine grid: H update ===
        st_f = FDTDState(ex=ex_f, ey=ey_f, ez=ez_f,
                         hx=hx_f, hy=hy_f, hz=hz_f,
                         step=jnp.array(step, dtype=jnp.int32))
        st_f = update_h(st_f, mats_f, dt, dx_f)

        # === Coarse grid: E update + CPML + PEC ===
        st_c = update_e(st_c, mats_c, dt, dx_c)
        st_c, cpml_state = apply_cpml_e(st_c, cpml_params, cpml_state, grid_c, cpml_axes)
        st_c = apply_pec(st_c)
        if pec_mask_c is not None:
            st_c = apply_pec_mask(st_c, pec_mask_c)

        # === Fine grid: E update + PEC mask ===
        st_f = update_e(st_f, mats_f, dt, dx_f)
        if pec_mask_f is not None:
            st_f = apply_pec_mask(st_f, pec_mask_f)

        # === Shared-node coupling ===
        (ex_c, ey_c, ez_c), (ex_f, ey_f, ez_f) = _shared_node_coupling_3d(
            (st_c.ex, st_c.ey, st_c.ez),
            (st_f.ex, st_f.ey, st_f.ez),
            subgrid_config,
        )
        hx_c, hy_c, hz_c = st_c.hx, st_c.hy, st_c.hz
        hx_f, hy_f, hz_f = st_f.hx, st_f.hy, st_f.hz

        # === Source injection on fine grid ===
        for src_i, src_j, src_k, src_comp, src_waveform in sources_f:
            if src_comp == "ez":
                ez_f = ez_f.at[src_i, src_j, src_k].add(float(src_waveform[step]))
            elif src_comp == "ex":
                ex_f = ex_f.at[src_i, src_j, src_k].add(float(src_waveform[step]))
            elif src_comp == "ey":
                ey_f = ey_f.at[src_i, src_j, src_k].add(float(src_waveform[step]))

        # === Probe recording on fine grid ===
        for p_idx, (pi, pj, pk) in enumerate(probe_indices_f):
            comp = probe_components[p_idx]
            if comp == "ez":
                time_series[step, p_idx] = float(ez_f[pi, pj, pk])
            elif comp == "ex":
                time_series[step, p_idx] = float(ex_f[pi, pj, pk])
            elif comp == "ey":
                time_series[step, p_idx] = float(ey_f[pi, pj, pk])

    # Final states
    final_c = FDTDState(ex=ex_c, ey=ey_c, ez=ez_c,
                        hx=hx_c, hy=hy_c, hz=hz_c,
                        step=jnp.array(n_steps, dtype=jnp.int32))
    final_f = FDTDState(ex=ex_f, ey=ey_f, ez=ez_f,
                        hx=hx_f, hy=hy_f, hz=hz_f,
                        step=jnp.array(n_steps, dtype=jnp.int32))

    return {
        "state_c": final_c,
        "state_f": final_f,
        "time_series": jnp.array(time_series),
        "config": subgrid_config,
        "dt": dt,
    }
