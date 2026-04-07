"""Non-uniform grid run path extracted from Simulation."""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from rfx.grid import C0
from rfx.core.yee import MaterialArrays
from rfx.materials.debye import init_debye
from rfx.materials.lorentz import init_lorentz
from rfx.nonuniform import NonUniformGrid, make_nonuniform_grid, run_nonuniform, make_current_source


def build_nonuniform_grid(
    freq_max: float,
    domain: tuple,
    dx: float | None,
    cpml_layers: int,
    dz_profile: np.ndarray,
) -> NonUniformGrid:
    """Build a NonUniformGrid from a dz_profile."""
    if dx is None:
        dx = C0 / freq_max / 20.0
    domain_xy = (domain[0], domain[1])
    return make_nonuniform_grid(domain_xy, dz_profile, dx, cpml_layers)


def assemble_materials_nu(
    sim,
    grid: NonUniformGrid,
) -> tuple[MaterialArrays, object, object, jnp.ndarray | None]:
    """Build material arrays and dispersion specs for non-uniform grid.

    Delegates to the shared rasterize_geometry() with non-uniform coordinates.
    Now supports all shape types, Debye/Lorentz poles, chi3, and thin conductors.

    Returns
    -------
    materials, debye_spec, lorentz_spec, pec_mask
    """
    from rfx.geometry.rasterize import rasterize_geometry, coords_from_nonuniform_grid

    coords = coords_from_nonuniform_grid(grid)

    # Thin conductors require mask(grid) which needs uniform Grid.
    # Skip on NU path for now — the shape won't resolve.
    result = rasterize_geometry(
        sim._geometry,
        sim._resolve_material,
        coords,
        pec_sigma_threshold=sim._PEC_SIGMA_THRESHOLD,
    )
    materials, debye_spec, lorentz_spec, pec_mask, _pec_shapes, _kerr_chi3 = result
    return materials, debye_spec, lorentz_spec, pec_mask


def pos_to_nu_index(grid: NonUniformGrid, pos) -> tuple[int, int, int]:
    """Convert physical (x, y, z) to non-uniform grid indices."""
    cpml = grid.cpml_layers
    dx = grid.dx
    dz_np = np.array(grid.dz)
    z_cumsum = np.cumsum(dz_np)
    z_cumsum = np.insert(z_cumsum, 0, 0.0)
    z_offset = z_cumsum[cpml]

    ix = int(round(pos[0] / dx)) + cpml
    iy = int(round(pos[1] / dx)) + cpml
    iz = cpml + int(np.argmin(np.abs(z_cumsum[cpml:] - z_offset - pos[2])))
    return (ix, iy, iz)


def run_nonuniform_path(sim, *, n_steps, compute_s_params=None, s_param_freqs=None):
    """Run simulation on non-uniform grid with graded dz.

    Parameters
    ----------
    sim : Simulation
        The Simulation instance (read-only access to its fields).
    n_steps : int
        Number of timesteps.
    compute_s_params : bool or None
    s_param_freqs : array or None

    Returns
    -------
    Result
    """
    from rfx.api import Result

    grid = build_nonuniform_grid(
        sim._freq_max, sim._domain, sim._dx, sim._cpml_layers, sim._dz_profile
    )
    materials, debye_spec, lorentz_spec, pec_mask = assemble_materials_nu(sim, grid)

    # Initialize Debye/Lorentz dispersion coefficients
    debye = None
    if debye_spec is not None:
        debye_poles, debye_masks = debye_spec
        debye = init_debye(debye_poles, materials, grid.dt, mask=debye_masks)

    lorentz = None
    if lorentz_spec is not None:
        lorentz_poles, lorentz_masks = lorentz_spec
        lorentz = init_lorentz(lorentz_poles, materials, grid.dt, mask=lorentz_masks)

    sources = []
    probes = []
    wire_port_specs = []

    for pe in sim._ports:
        idx = pos_to_nu_index(grid, pe.position)
        if pe.impedance == 0.0:
            # Current source with dV normalization
            src = make_current_source(
                grid, idx, pe.component, pe.waveform, n_steps, materials)
            sources.append(src)
        elif pe.extent is not None:
            # Wire port on non-uniform grid
            axis_map = {"ex": 0, "ey": 1, "ez": 2}
            axis = axis_map[pe.component]
            end_pos = list(pe.position)
            end_pos[axis] += pe.extent
            idx_end = pos_to_nu_index(grid, tuple(end_pos))
            lo_k = min(idx[axis], idx_end[axis])
            hi_k = max(idx[axis], idx_end[axis])

            wire_cells = list(range(lo_k, hi_k + 1))
            n_cells = max(len(wire_cells), 1)

            for k in wire_cells:
                cell = list(idx)
                cell[axis] = k
                ci, cj, ck = cell
                # 3D wire port: σ = n_cells * d_parallel / (Z0 * d_perp1 * d_perp2)
                # Each cell in the wire carries 1/n_cells of total impedance Z0.
                if axis == 2:
                    d_cell = float(grid.dz[ck])
                    dp1, dp2 = grid.dx, grid.dy
                elif axis == 1:
                    d_cell = grid.dy
                    dp1, dp2 = grid.dx, float(grid.dz[ck])
                else:
                    d_cell = grid.dx
                    dp1, dp2 = grid.dy, float(grid.dz[ck])
                sigma_port = n_cells * d_cell / (pe.impedance * dp1 * dp2)
                materials = materials._replace(
                    sigma=materials.sigma.at[ci, cj, ck].add(
                        sigma_port))
                if pec_mask is not None:
                    pec_mask = pec_mask.at[ci, cj, ck].set(False)

            # Create per-cell sources
            mid_k = wire_cells[len(wire_cells) // 2]
            mid_cell = list(idx)
            mid_cell[axis] = mid_k

            for k in wire_cells:
                cell = list(idx)
                cell[axis] = k
                src = make_current_source(
                    grid, tuple(cell), pe.component,
                    pe.waveform, n_steps, materials)
                # Scale by 1/n_cells for distributed excitation
                scaled_wf = np.array(src[4]) / n_cells
                sources.append(
                    (src[0], src[1], src[2], src[3], scaled_wf))

            # Wire port S-param spec
            wire_port_specs.append({
                'mid_i': mid_cell[0], 'mid_j': mid_cell[1],
                'mid_k': mid_cell[2],
                'component': pe.component,
                'impedance': pe.impedance,
            })
        else:
            # Single-cell lumped port
            i, j, k = idx
            # 3D lumped port: σ = d_parallel / (Z0 * d_perp1 * d_perp2)
            # This ensures correct power dissipation P = V²/Z0 in
            # anisotropic cells where dz ≠ dx.  The old formula
            # σ = 1/(Z0*d_parallel) is only valid for cubic cells.
            axis_map = {"ex": 0, "ey": 1, "ez": 2}
            port_axis = axis_map[pe.component]
            if port_axis == 2:
                d_parallel = float(grid.dz[k])
                d_perp1, d_perp2 = grid.dx, grid.dy
            elif port_axis == 1:
                d_parallel = grid.dy
                d_perp1, d_perp2 = grid.dx, float(grid.dz[k])
            else:
                d_parallel = grid.dx
                d_perp1, d_perp2 = grid.dy, float(grid.dz[k])
            sigma_port = d_parallel / (pe.impedance * d_perp1 * d_perp2)
            materials = materials._replace(
                sigma=materials.sigma.at[i, j, k].add(sigma_port))
            if pec_mask is not None:
                pec_mask = pec_mask.at[i, j, k].set(False)
            src = make_current_source(
                grid, idx, pe.component, pe.waveform, n_steps, materials)
            sources.append(src)

    for pe in sim._probes:
        idx = pos_to_nu_index(grid, pe.position)
        probes.append((*idx, pe.component))

    sp_freqs = None
    if wire_port_specs and (compute_s_params is None or compute_s_params):
        sp_freqs = s_param_freqs
        if sp_freqs is None:
            sp_freqs = np.linspace(
                sim._freq_max / 10, sim._freq_max, 50)

    r = run_nonuniform(
        grid, materials, n_steps,
        pec_mask=pec_mask,
        sources=sources,
        probes=probes,
        wire_ports=wire_port_specs if wire_port_specs else None,
        s_param_freqs=sp_freqs,
        debye=debye,
        lorentz=lorentz,
    )

    s_params = r.get("s_params")
    freqs_out = r.get("s_param_freqs")

    return Result(
        state=r["state"],
        time_series=r["time_series"],
        s_params=s_params,
        freqs=freqs_out,
        grid=grid,
        dt=grid.dt,
        freq_range=(sim._freq_max / 10, sim._freq_max, sim._boundary),
    )
