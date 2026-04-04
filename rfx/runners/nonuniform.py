"""Non-uniform grid run path extracted from Simulation."""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from rfx.grid import C0
from rfx.core.yee import MaterialArrays
from rfx.materials.debye import DebyePole, init_debye
from rfx.materials.lorentz import LorentzPole, init_lorentz
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

    Returns
    -------
    materials : MaterialArrays
    debye_spec : (poles, masks) or None
    lorentz_spec : (poles, masks) or None
    pec_mask : bool array or None
    """
    shape = (grid.nx, grid.ny, grid.nz)
    eps_r = jnp.ones(shape, dtype=jnp.float32)
    sigma = jnp.zeros(shape, dtype=jnp.float32)
    mu_r = jnp.ones(shape, dtype=jnp.float32)
    pec_mask = jnp.zeros(shape, dtype=jnp.bool_)

    cpml = grid.cpml_layers
    dx = grid.dx
    # Cumulative z positions for index mapping
    dz_np = np.array(grid.dz)
    z_cumsum = np.cumsum(dz_np)
    z_cumsum = np.insert(z_cumsum, 0, 0.0)

    PEC_SIGMA_THRESHOLD = sim._PEC_SIGMA_THRESHOLD

    # Per-pole dispersion masks (same approach as uniform _assemble_materials)
    debye_masks_by_pole: dict[DebyePole, jnp.ndarray] = {}
    lorentz_masks_by_pole: dict[LorentzPole, jnp.ndarray] = {}

    for entry in sim._geometry:
        mat = sim._resolve_material(entry.material_name)
        shape_obj = entry.shape
        if hasattr(shape_obj, 'corner_lo') and hasattr(shape_obj, 'corner_hi'):
            c1, c2 = shape_obj.corner_lo, shape_obj.corner_hi
            # x,y: uniform grid mapping (physical coords include CPML offset)
            ix0 = max(0, int(round(c1[0] / dx)) + cpml)
            ix1 = min(grid.nx, int(round(c2[0] / dx)) + cpml)
            iy0 = max(0, int(round(c1[1] / dx)) + cpml)
            iy1 = min(grid.ny, int(round(c2[1] / dx)) + cpml)
            # z: map physical z to non-uniform grid index
            z_lo_phys = c1[2]
            z_hi_phys = c2[2]
            # z_cumsum[cpml] = start of physical domain
            z_offset = z_cumsum[cpml]
            iz0 = cpml + int(np.argmin(np.abs(
                z_cumsum[cpml:] - z_offset - z_lo_phys)))
            iz1 = cpml + int(np.argmin(np.abs(
                z_cumsum[cpml:] - z_offset - z_hi_phys)))
            if iz1 <= iz0:
                iz1 = iz0 + 1  # at least 1 cell

            if ix0 < ix1 and iy0 < iy1 and iz0 < iz1:
                if mat.sigma >= PEC_SIGMA_THRESHOLD:
                    pec_mask = pec_mask.at[ix0:ix1, iy0:iy1, iz0:iz1].set(True)
                else:
                    eps_r = eps_r.at[ix0:ix1, iy0:iy1, iz0:iz1].set(mat.eps_r)
                    sigma = sigma.at[ix0:ix1, iy0:iy1, iz0:iz1].set(mat.sigma)
                    mu_r = mu_r.at[ix0:ix1, iy0:iy1, iz0:iz1].set(mat.mu_r)

                # Build spatial mask for this box (for dispersion pole lookup)
                if mat.debye_poles or mat.lorentz_poles:
                    box_mask = jnp.zeros(shape, dtype=jnp.bool_)
                    box_mask = box_mask.at[ix0:ix1, iy0:iy1, iz0:iz1].set(True)

                    if mat.debye_poles:
                        for pole in mat.debye_poles:
                            if pole in debye_masks_by_pole:
                                debye_masks_by_pole[pole] = debye_masks_by_pole[pole] | box_mask
                            else:
                                debye_masks_by_pole[pole] = box_mask

                    if mat.lorentz_poles:
                        for pole in mat.lorentz_poles:
                            if pole in lorentz_masks_by_pole:
                                lorentz_masks_by_pole[pole] = lorentz_masks_by_pole[pole] | box_mask
                            else:
                                lorentz_masks_by_pole[pole] = box_mask

    materials = MaterialArrays(eps_r=eps_r, sigma=sigma, mu_r=mu_r)

    debye_spec = None
    if debye_masks_by_pole:
        debye_poles = list(debye_masks_by_pole)
        debye_masks = [debye_masks_by_pole[pole] for pole in debye_poles]
        debye_spec = (debye_poles, debye_masks)

    lorentz_spec = None
    if lorentz_masks_by_pole:
        lorentz_poles = list(lorentz_masks_by_pole)
        lorentz_masks = [lorentz_masks_by_pole[pole] for pole in lorentz_poles]
        lorentz_spec = (lorentz_poles, lorentz_masks)

    has_pec = bool(jnp.any(pec_mask))
    return materials, debye_spec, lorentz_spec, pec_mask if has_pec else None


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
                # Port impedance loading: sigma = n_cells / (Z0 * d_parallel)
                # For z-directed port, d_parallel = dz[k]; for x/y, use dx/dy
                if axis == 2:
                    d_cell = float(grid.dz[ck])
                elif axis == 1:
                    d_cell = grid.dy
                else:
                    d_cell = grid.dx
                sigma_port = n_cells / (pe.impedance * d_cell)
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
            # Use d_parallel for the port component direction
            axis_map = {"ex": 0, "ey": 1, "ez": 2}
            port_axis = axis_map[pe.component]
            if port_axis == 2:
                d_port = float(grid.dz[k])
            elif port_axis == 1:
                d_port = grid.dy
            else:
                d_port = grid.dx
            sigma_port = 1.0 / (pe.impedance * d_port)
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
        dt=grid.dt,
        freq_range=(sim._freq_max / 10, sim._freq_max, sim._boundary),
    )
