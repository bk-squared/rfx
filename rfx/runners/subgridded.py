"""Subgridded (SBP-SAT) run path extracted from Simulation."""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

from rfx.grid import Grid
from rfx.subgridding.sbp_sat_3d import phase1_3d_dt


def run_subgridded_path(sim, grid_coarse, base_materials_coarse, pec_mask_coarse,
                        n_steps):
    """Run the canonical Phase-1 z-slab SBP-SAT path (JIT-compiled).

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
    from rfx.subgridding.face_ops import build_zface_ops
    from rfx.subgridding.sbp_sat_3d import SubgridConfig3D
    from rfx.subgridding.jit_runner import run_subgridded_jit as _run_sg

    if hasattr(sim, "_validate_phase1_subgrid_boundaries"):
        sim._validate_phase1_subgrid_boundaries()
    elif sim._boundary != "pec":
        raise ValueError(
            "Phase-1 SBP-SAT z-slab subgridding supports boundary='pec' only"
        )
    if getattr(sim, "_coaxial_ports", None):
        raise ValueError(
            "Phase-1 SBP-SAT z-slab subgridding does not support coaxial ports"
        )
    if any(pe.impedance != 0.0 or pe.extent is not None for pe in sim._ports):
        raise ValueError(
            "Phase-1 SBP-SAT z-slab subgridding supports soft point sources only; "
            "impedance point ports and wire/extent ports are deferred"
        )

    ref = sim._refinement
    ratio = ref["ratio"]
    z_lo, z_hi = ref["z_range"]
    tau = ref.get("tau", 0.5)
    dx_c = grid_coarse.dx
    dx_f = dx_c / ratio
    if ref.get("xy_margin") is not None:
        raise ValueError(
            "Phase-1 SBP-SAT z-slab subgridding does not support xy_margin"
        )

    fk_lo = max(int(round(z_lo / dx_c)), 0)
    fk_hi = min(int(round(z_hi / dx_c)) + 1, grid_coarse.nz)
    if fk_hi <= fk_lo:
        raise ValueError(f"z_range={ref['z_range']} maps to an empty coarse z slab")

    # Phase 1: fine region spans the full supported x/y interior.
    fi_lo = 0
    fi_hi = grid_coarse.nx
    fj_lo = 0
    fj_hi = grid_coarse.ny

    nx_f = (fi_hi - fi_lo) * ratio
    ny_f = (fj_hi - fj_lo) * ratio
    nz_f = (fk_hi - fk_lo) * ratio

    dt = phase1_3d_dt(dx_f)

    config = SubgridConfig3D(
        nx_c=grid_coarse.nx, ny_c=grid_coarse.ny, nz_c=grid_coarse.nz,
        dx_c=dx_c,
        fi_lo=fi_lo, fi_hi=fi_hi,
        fj_lo=fj_lo, fj_hi=fj_hi,
        fk_lo=fk_lo, fk_hi=fk_hi,
        nx_f=nx_f, ny_f=ny_f, nz_f=nz_f,
        dx_f=dx_f, dt=float(dt), ratio=ratio, tau=tau,
        face_ops=build_zface_ops((fi_hi - fi_lo, fj_hi - fj_lo), ratio, dx_c),
    )

    overlap = (slice(fi_lo, fi_hi), slice(fj_lo, fj_hi), slice(fk_lo, fk_hi))
    mats_c = base_materials_coarse._replace(
        eps_r=base_materials_coarse.eps_r.at[overlap].set(1.0),
        sigma=base_materials_coarse.sigma.at[overlap].set(0.0),
        mu_r=base_materials_coarse.mu_r.at[overlap].set(1.0),
    )
    pec_mask_c = pec_mask_coarse
    if pec_mask_c is not None:
        pec_mask_c = pec_mask_c.at[overlap].set(False)

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
    x_off = fi_lo * dx_c
    y_off = fj_lo * dx_c
    z_off = fk_lo * dx_c

    from rfx.geometry.rasterize import coords_from_fine_grid, rasterize_geometry

    coords_f = coords_from_fine_grid(nx_f, ny_f, nz_f, dx_f, x_off, y_off, z_off)
    mats_f, _, _, pec_mask_f, _, _ = rasterize_geometry(
        sim._geometry,
        sim._resolve_material,
        coords_f,
        pec_sigma_threshold=sim._PEC_SIGMA_THRESHOLD,
    )
    has_pec_f = bool(jnp.any(pec_mask_f)) if pec_mask_f is not None else False

    def _pos_to_fine_idx(pos):
        idx = (
            int(round((pos[0] - x_off) / dx_f)),
            int(round((pos[1] - y_off) / dx_f)),
            int(round((pos[2] - z_off) / dx_f)),
        )
        if not (0 <= idx[0] < nx_f and 0 <= idx[1] < ny_f and 0 <= idx[2] < nz_f):
            raise ValueError(
                f"Position {pos} maps to fine-grid index {idx} outside "
                f"the Phase-1 z-slab fine grid shape ({nx_f}, {ny_f}, {nz_f}). "
                "Widen z_range to cover all sources and probes."
            )
        return idx

    # Build sources on fine grid
    sources_f = []
    times = jnp.arange(n_steps, dtype=jnp.float32) * dt

    for pe in sim._ports:
        # Phase 1 supports soft point sources only; impedance and wire
        # ports are rejected before this runner is entered.
        idx = _pos_to_fine_idx(pe.position)
        i, j, k = idx
        waveform = jax.vmap(pe.waveform)(times)
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
        mats_c,
        mats_f,
        config,
        n_steps,
        pec_mask_c=pec_mask_c,
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
        freq_range=(sim._freq_max / 10, sim._freq_max, sim._boundary),
    )
