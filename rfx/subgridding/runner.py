"""Subgridded simulation runner.

Runs a coupled coarse+fine FDTD simulation using SBP-SAT subgridding.
CPML is applied on the coarse grid boundaries. Sources and probes
operate on the fine grid (where the structure of interest resides).
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from rfx.core.yee import FDTDState, MaterialArrays
from rfx.stepping import update_h, update_e
from rfx.boundaries.pec import apply_pec, apply_pec_mask
# Patching rfx.boundaries.cpml.init_cpml later does not replace this bound import.
from rfx.boundaries.cpml import init_cpml, apply_cpml_h, apply_cpml_e
from rfx.grid import Grid
from rfx.geometry.rasterize_grid import unpadded_high_walls, fine_high_walls
from rfx.model.materials import with_components
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
    high_wall_c = unpadded_high_walls(grid_c)
    high_wall_f = fine_high_walls(grid_c, subgrid_config)
    periodic = tuple(a in getattr(grid_c, "periodic_axes", "") for a in "xyz")
    mats_c = with_components(mats_c, grid_c, periodic=periodic, high_wall=high_wall_c)
    mats_f = with_components(mats_f, None, periodic=(False,) * 3, high_wall=high_wall_f)

    # Override coarse grid dt to match the subgrid global timestep
    # (fine grid CFL is more restrictive than coarse grid CFL)
    grid_c.dt = dt

    # Initialize CPML on coarse grid
    cpml_params, cpml_state = init_cpml(grid_c)

    # Initialize field states
    shape_c = (subgrid_config.nx_c, subgrid_config.ny_c, subgrid_config.nz_c)
    shape_f = (subgrid_config.nx_f, subgrid_config.ny_f, subgrid_config.nz_f)

    def z(s):
        return jnp.zeros(s, dtype=jnp.float32)
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

    ex_c, ex_f, ey_c, ey_f, ez_c, ez_f, hx_c, hx_f, hy_c, hy_f, hz_c, hz_f = run_subgridded_loop(
        _log_interval=_log_interval,
        _t0=_t0,
        _time=_time,
        cpml_axes=cpml_axes,
        cpml_params=cpml_params,
        cpml_state=cpml_state,
        dt=dt,
        dx_c=dx_c,
        dx_f=dx_f,
        ex_c=ex_c,
        ex_f=ex_f,
        ey_c=ey_c,
        ey_f=ey_f,
        ez_c=ez_c,
        ez_f=ez_f,
        grid_c=grid_c,
        high_wall_c=high_wall_c,
        high_wall_f=high_wall_f,
        hx_c=hx_c,
        hx_f=hx_f,
        hy_c=hy_c,
        hy_f=hy_f,
        hz_c=hz_c,
        hz_f=hz_f,
        mats_c=mats_c,
        mats_f=mats_f,
        n_steps=n_steps,
        pec_mask_c=pec_mask_c,
        pec_mask_f=pec_mask_f,
        probe_components=probe_components,
        probe_indices_f=probe_indices_f,
        sources_f=sources_f,
        subgrid_config=subgrid_config,
        time_series=time_series,
    )

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


from rfx.stepping.subgrid_loop import run_subgridded_loop
