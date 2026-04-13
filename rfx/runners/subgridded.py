"""Subgridded (SBP-SAT) run path extracted from Simulation."""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

from rfx.core.yee import EPS_0, MU_0
from rfx.grid import Grid


def run_subgridded_path(sim, grid_coarse, base_materials_coarse, pec_mask_coarse,
                        n_steps):
    """Run simulation using SBP-SAT subgridding (JIT-compiled).

    Parameters
    ----------
    sim : Simulation
        The Simulation instance (read-only access to its fields).
    grid_coarse : Grid
        Coarse uniform grid.
    base_materials_coarse : MaterialArrays
        Material arrays on the coarse grid.
    pec_mask_coarse : jnp.ndarray or None
        PEC mask on the coarse grid.
    n_steps : int
        Number of timesteps.

    Returns
    -------
    Result
    """
    from rfx.api import Result
    from rfx.subgridding.sbp_sat_3d import SubgridConfig3D
    from rfx.subgridding.jit_runner import run_subgridded_jit as _run_sg

    ref = sim._refinement
    ratio = ref["ratio"]
    z_lo, z_hi = ref["z_range"]
    tau = ref.get("tau", 0.5)
    dx_c = grid_coarse.dx
    dx_f = dx_c / ratio
    xy_margin = ref["xy_margin"] if ref["xy_margin"] is not None else 2 * dx_c

    # Map z_range to coarse grid indices
    cpml = grid_coarse.cpml_layers
    fk_lo = max(int(round(z_lo / dx_c)) + cpml, cpml)
    fk_hi = min(int(round(z_hi / dx_c)) + cpml + 1, grid_coarse.nz - cpml)

    # Fine region covers full x,y for simplicity (with cpml margin)
    fi_lo = cpml
    fi_hi = grid_coarse.nx - cpml
    fj_lo = cpml
    fj_hi = grid_coarse.ny - cpml

    # Fine grid dimensions
    nx_f = (fi_hi - fi_lo) * ratio
    ny_f = (fj_hi - fj_lo) * ratio
    nz_f = (fk_hi - fk_lo) * ratio

    # Global timestep (limited by fine grid CFL)
    C0_val = 1.0 / np.sqrt(float(EPS_0) * float(MU_0))
    dt = 0.45 * dx_f / (C0_val * np.sqrt(3))

    config = SubgridConfig3D(
        nx_c=grid_coarse.nx, ny_c=grid_coarse.ny, nz_c=grid_coarse.nz,
        dx_c=dx_c,
        fi_lo=fi_lo, fi_hi=fi_hi,
        fj_lo=fj_lo, fj_hi=fj_hi,
        fk_lo=fk_lo, fk_hi=fk_hi,
        nx_f=nx_f, ny_f=ny_f, nz_f=nz_f,
        dx_f=dx_f, dt=float(dt), ratio=ratio, tau=tau,
    )

    # Build fine-grid materials by rasterizing geometry at fine resolution
    shape_f = (nx_f, ny_f, nz_f)

    # Create a Grid for fine region (for position_to_index utility)
    fine_domain = (nx_f * dx_f, ny_f * dx_f, nz_f * dx_f)
    fine_grid = Grid(
        freq_max=sim._freq_max,
        domain=fine_domain,
        dx=dx_f,
        cpml_layers=0,
    )
    # Override shape to match exactly (Grid may add +1 rounding)
    fine_grid._shape_override = shape_f

    # Rasterize geometry into fine grid materials using shared function.
    # Uses cell-center coordinates (not cell edges) for correct placement.
    x_off = (fi_lo - cpml) * dx_c
    y_off = (fj_lo - cpml) * dx_c
    z_off = (fk_lo - cpml) * dx_c

    from rfx.geometry.rasterize import coords_from_fine_grid, rasterize_geometry

    coords_f = coords_from_fine_grid(nx_f, ny_f, nz_f, dx_f, x_off, y_off, z_off)
    mats_f, _, _, pec_mask_f, _, _ = rasterize_geometry(
        sim._geometry,
        sim._resolve_material,
        coords_f,
        pec_sigma_threshold=sim._PEC_SIGMA_THRESHOLD,
    )
    has_pec_f = bool(jnp.any(pec_mask_f)) if pec_mask_f is not None else False

    # Helper: convert physical position to fine-grid index
    def _pos_to_fine_idx(pos):
        idx = (
            int(round((pos[0] - x_off) / dx_f)),
            int(round((pos[1] - y_off) / dx_f)),
            int(round((pos[2] - z_off) / dx_f)),
        )
        # Bounds check — source/probe outside fine grid causes garbage results
        if not (0 <= idx[0] < nx_f and 0 <= idx[1] < ny_f and 0 <= idx[2] < nz_f):
            import warnings
            warnings.warn(
                f"Position {pos} maps to fine-grid index {idx} which is outside "
                f"the fine grid shape ({nx_f}, {ny_f}, {nz_f}). "
                f"Widen z_range in add_refinement() to cover all sources and probes.",
                stacklevel=3,
            )
        return idx

    # Build sources on fine grid
    sources_f = []
    times = jnp.arange(n_steps, dtype=jnp.float32) * dt

    for pe in sim._ports:
        axis_map = {"ex": 0, "ey": 1, "ez": 2}
        axis = axis_map[pe.component]

        if pe.impedance == 0.0:
            # Soft source — normalization depends on boundary type:
            # PEC: raw field add (matches make_source in uniform runner)
            # CPML/UPML: J-source Cb normalized (matches make_j_source)
            idx = _pos_to_fine_idx(pe.position)
            i, j, k = idx
            raw_waveform = jax.vmap(pe.waveform)(times)
            if sim._boundary in ("cpml", "upml"):
                eps = float(mats_f.eps_r[i, j, k]) * EPS_0
                sigma_val = float(mats_f.sigma[i, j, k])
                loss = sigma_val * dt / (2.0 * eps)
                cb = (dt / eps) / (1.0 + loss)
                waveform = cb * raw_waveform
            else:
                waveform = raw_waveform
            sources_f.append((i, j, k, pe.component, np.array(waveform)))
            continue

        if pe.extent is not None:
            # Wire port: compute cells manually
            idx_start = _pos_to_fine_idx(pe.position)
            end_pos = list(pe.position)
            end_pos[axis] += pe.extent
            idx_end = _pos_to_fine_idx(tuple(end_pos))

            lo = min(idx_start[axis], idx_end[axis])
            hi = max(idx_start[axis], idx_end[axis])
            cells = []
            for a in range(lo, hi + 1):
                cell = list(idx_start)
                cell[axis] = a
                cells.append(tuple(cell))

            n_cells = max(len(cells), 1)
            # Distribute port impedance
            sigma_port_per_cell = n_cells / (pe.impedance * dx_f)
            for cell in cells:
                i, j, k = cell
                mats_f = mats_f._replace(
                    sigma=mats_f.sigma.at[i, j, k].add(sigma_port_per_cell))
                pec_mask_f = pec_mask_f.at[i, j, k].set(False)

            # Precompute Cb-corrected waveforms
            for cell in cells:
                i, j, k = cell
                eps = float(mats_f.eps_r[i, j, k]) * EPS_0
                sigma_val = float(mats_f.sigma[i, j, k])
                loss = sigma_val * dt / (2.0 * eps)
                cb = (dt / eps) / (1.0 + loss)
                waveform = (cb / dx_f) * jax.vmap(pe.waveform)(times) / n_cells
                sources_f.append((i, j, k, pe.component, np.array(waveform)))
        else:
            # Lumped port
            idx = _pos_to_fine_idx(pe.position)
            i, j, k = idx
            sigma_port = 1.0 / (pe.impedance * dx_f)
            mats_f = mats_f._replace(
                sigma=mats_f.sigma.at[i, j, k].add(sigma_port))
            pec_mask_f = pec_mask_f.at[i, j, k].set(False)

            eps = float(mats_f.eps_r[i, j, k]) * EPS_0
            sigma_val = float(mats_f.sigma[i, j, k])
            loss = sigma_val * dt / (2.0 * eps)
            cb = (dt / eps) / (1.0 + loss)
            waveform = (cb / dx_f) * jax.vmap(pe.waveform)(times)
            sources_f.append((i, j, k, pe.component, np.array(waveform)))

    # Build probes on fine grid
    probe_indices_f = []
    probe_components = []
    for pe in sim._probes:
        idx = _pos_to_fine_idx(pe.position)
        probe_indices_f.append(idx)
        probe_components.append(pe.component)

    result = _run_sg(
        grid_coarse,
        base_materials_coarse,
        mats_f,
        config,
        n_steps,
        pec_mask_c=pec_mask_coarse,
        pec_mask_f=pec_mask_f if has_pec_f else None,
        sources_f=sources_f,
        probe_indices_f=probe_indices_f,
        probe_components=probe_components,
    )

    return Result(
        state=result.state_f,
        time_series=result.time_series,
        s_params=None,
        freqs=None,
        grid=fine_grid,
        dt=dt,
        freq_range=(sim._freq_max / 10, sim._freq_max, 'cpml'),
    )
