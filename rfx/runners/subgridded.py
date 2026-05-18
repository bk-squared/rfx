"""Subgridded (SBP-SAT) run path extracted from Simulation."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import jax
import jax.numpy as jnp

from rfx.core.yee import EPS_0, MU_0
from rfx.grid import Grid


def _run_subgridded_once(
    sim,
    grid_coarse,
    base_materials_coarse,
    pec_mask_coarse,
    n_steps,
    *,
    diagnostic_lumped_sparam_freqs_override=None,
    diagnostic_lumped_sparam_driven_index_override=None,
):
    """Run one SBP-SAT subgrid simulation, optionally collecting one S-column.

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
    from rfx.subgridding.jit_runner import (
        SubgridRunOptions,
        run_subgridded_jit as _run_sg,
    )

    ref = sim._refinement
    ratio = ref["ratio"]
    z_lo, z_hi = ref["z_range"]
    tau = ref.get("tau", 0.5)
    dx_c = grid_coarse.dx
    dx_f = dx_c / ratio

    # Map z_range to coarse grid indices
    pad_z_lo = int(getattr(grid_coarse, "pad_z_lo", grid_coarse.cpml_layers))
    pad_z_hi = int(getattr(grid_coarse, "pad_z_hi", grid_coarse.cpml_layers))
    fk_lo = max(int(round(z_lo / dx_c)) + pad_z_lo, pad_z_lo)
    fk_hi = min(int(round(z_hi / dx_c)) + pad_z_lo + 1, grid_coarse.nz - pad_z_hi)

    # Fine region covers full x/y by default.  A finite xy_margin enables a
    # research-only local x/y window inset from the physical x/y boundaries;
    # production validation rejects that lane until its waveform gates pass.
    xy_margin = ref.get("xy_margin")
    if xy_margin is None:
        fi_lo = grid_coarse.pad_x_lo
        fi_hi = grid_coarse.nx - grid_coarse.pad_x_hi
        fj_lo = grid_coarse.pad_y_lo
        fj_hi = grid_coarse.ny - grid_coarse.pad_y_hi
    else:
        margin = float(xy_margin)
        fi_lo = max(
            int(round(margin / dx_c)) + grid_coarse.pad_x_lo,
            grid_coarse.pad_x_lo,
        )
        fi_hi = min(
            int(round((sim._domain[0] - margin) / dx_c))
            + grid_coarse.pad_x_lo
            + 1,
            grid_coarse.nx - grid_coarse.pad_x_hi,
        )
        fj_lo = max(
            int(round(margin / dx_c)) + grid_coarse.pad_y_lo,
            grid_coarse.pad_y_lo,
        )
        fj_hi = min(
            int(round((sim._domain[1] - margin) / dx_c))
            + grid_coarse.pad_y_lo
            + 1,
            grid_coarse.ny - grid_coarse.pad_y_hi,
        )

    # Fine grid dimensions.
    #
    # Grid.shape uses endpoint nodes: a physical span with N coarse cells has
    # N+1 coarse indices.  A node-aligned fine slab over the same span must
    # therefore have ``(N * ratio) + 1`` fine indices, or equivalently
    # ``(n_coarse_nodes - 1) * ratio + 1``.  The legacy ``n_nodes * ratio``
    # formula created one extra fine interval per axis and made the z-hi fine
    # face physically offset from the coarse face it was SAT-coupled to.
    #
    # ``overlap_fine_extent`` is the single source of truth shared with
    # ``rfx.subgridding.validation.build_subgrid_region`` so the validation
    # report and this executed fine grid cannot drift apart.
    from rfx.subgridding.validation import overlap_fine_extent

    nx_f = overlap_fine_extent(fi_hi - fi_lo, ratio)
    ny_f = overlap_fine_extent(fj_hi - fj_lo, ratio)
    nz_f = overlap_fine_extent(fk_hi - fk_lo, ratio)

    # Global timestep (limited by fine grid CFL)
    C0_val = 1.0 / np.sqrt(float(EPS_0) * float(MU_0))
    # Use the same CFL factor as Grid.courant_dt (0.99/sqrt(3))
    # to match uniform runner's timestep for equivalent dx
    dt = 0.99 * dx_f / (C0_val * np.sqrt(3))

    config = SubgridConfig3D(
        nx_c=grid_coarse.nx, ny_c=grid_coarse.ny, nz_c=grid_coarse.nz,
        dx_c=dx_c,
        fi_lo=fi_lo, fi_hi=fi_hi,
        fj_lo=fj_lo, fj_hi=fj_hi,
        fk_lo=fk_lo, fk_hi=fk_hi,
        nx_f=nx_f, ny_f=ny_f, nz_f=nz_f,
        dx_f=dx_f, dt=float(dt), ratio=ratio, tau=tau,
    )

    validation_mode = ref.get("validation", "production")
    if validation_mode != "off":
        from rfx.subgridding.validation import validate_subgrid_setup
        validation_report = validate_subgrid_setup(
            sim,
            grid_coarse,
            base_materials_coarse,
            pec_mask_coarse,
            mode=validation_mode,
        )
        if validation_mode == "production":
            validation_report.raise_if_unsupported()

    topology = ref.get("topology", "overlap_z_slab")
    if topology != "overlap_z_slab":
        from rfx.runners.disjoint import run_disjoint_stage2_path

        return run_disjoint_stage2_path(sim, grid_coarse, n_steps)

    is_full_xy_region = (
        config.fi_lo == grid_coarse.pad_x_lo
        and config.fi_hi == grid_coarse.nx - grid_coarse.pad_x_hi
        and config.fj_lo == grid_coarse.pad_y_lo
        and config.fj_hi == grid_coarse.ny - grid_coarse.pad_y_hi
    )
    # The boundary-terminated exterior z-interface path is implemented only
    # for full-x/y z slabs.  Local x/y windows use the endpoint-node 6-face box
    # SAT path plus fine physical PEC faces; auto-selecting the z-slab-only
    # exterior path would let production validation pass but make execution
    # fail before the first timestep.
    auto_boundary_terminated_exterior = (
        validation_mode == "production"
        and is_full_xy_region
        and (
            (
                config.fk_lo <= int(getattr(grid_coarse, "pad_z_lo", 0))
                and "z_lo" in getattr(grid_coarse, "pec_faces", set())
            )
            ^ (
                config.fk_hi
                >= grid_coarse.nz - int(getattr(grid_coarse, "pad_z_hi", 0))
                and "z_hi" in getattr(grid_coarse, "pec_faces", set())
            )
            or (
                int(grid_coarse.cpml_layers) == 0
                and ((config.fk_lo <= 0) ^ (config.fk_hi >= config.nz_c))
            )
        )
    )

    # Build fine-grid materials by rasterizing geometry at fine resolution
    shape_f = (nx_f, ny_f, nz_f)

    # Create a Grid for fine region (for position_to_index utility)
    fine_domain = ((nx_f - 1) * dx_f, (ny_f - 1) * dx_f, (nz_f - 1) * dx_f)
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
    x_off = (fi_lo - grid_coarse.pad_x_lo) * dx_c
    y_off = (fj_lo - grid_coarse.pad_y_lo) * dx_c
    z_off = (fk_lo - grid_coarse.pad_z_lo) * dx_c

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
        # Bounds check — source/probe outside fine grid causes garbage results.
        # JAX scatter/gather can otherwise silently drop or clamp OOB indices,
        # so fail at the API boundary instead of returning a misleading run.
        if not (0 <= idx[0] < nx_f and 0 <= idx[1] < ny_f and 0 <= idx[2] < nz_f):
            raise ValueError(
                f"Position {pos} maps to fine-grid index {idx} which is outside "
                f"the fine grid shape ({nx_f}, {ny_f}, {nz_f}). "
                f"Widen z_range in add_refinement() to cover all sources and probes.",
            )
        return idx

    # Build sources on fine grid
    sources_f = []
    sources_c = []
    times = jnp.arange(n_steps, dtype=jnp.float32) * dt
    # Local x/y windows keep an overlapping coarse shadow grid.  The validated
    # central-source lane therefore co-injects soft sources on that coarse
    # shadow by default; explicit source/projection knobs remain diagnostic
    # overrides and are still rejected by production validation.
    inject_sources_on_coarse_shadow = bool(
        ref.get("inject_sources_on_coarse_shadow", ref.get("xy_margin") is not None)
    )
    coarse_shadow_source_scale = float(ref.get("coarse_shadow_source_scale", 1.0))
    fine_source_scale = float(ref.get("fine_source_scale", 1.0))
    coarse_shadow_source_projection = str(
        ref.get("coarse_shadow_source_projection", "physical_nearest")
    )

    def _fine_idx_to_position(idx):
        return (
            x_off + idx[0] * dx_f,
            y_off + idx[1] * dx_f,
            z_off + idx[2] * dx_f,
        )

    def _coarse_trilinear_entries(pos, component, waveform):
        coords = (
            pos[0] / dx_c + grid_coarse.pad_x_lo,
            pos[1] / dx_c + grid_coarse.pad_y_lo,
            (
                0.0
                if grid_coarse.is_2d
                else pos[2] / dx_c + grid_coarse.pad_z_lo
            ),
        )
        bounds = (grid_coarse.nx, grid_coarse.ny, grid_coarse.nz)
        axes = []
        for coord, upper in zip(coords, bounds):
            lo = int(np.floor(coord))
            lo = max(0, min(lo, upper - 1))
            hi = max(0, min(lo + 1, upper - 1))
            frac = float(coord - lo)
            if hi == lo:
                axes.append(((lo, 1.0),))
            else:
                axes.append(((lo, 1.0 - frac), (hi, frac)))
        entries = []
        for ci, wi in axes[0]:
            for cj, wj in axes[1]:
                for ck, wk in axes[2]:
                    weight = wi * wj * wk
                    if abs(weight) > 0.0:
                        entries.append(
                            (
                                ci,
                                cj,
                                ck,
                                component,
                                np.array(waveform) * coarse_shadow_source_scale * weight,
                            )
                        )
        return entries

    def _coarse_shadow_source_entries(pos, fine_idx, component, waveform):
        if coarse_shadow_source_projection == "physical_nearest":
            ci, cj, ck = grid_coarse.position_to_index(pos)
            return [
                (
                    ci,
                    cj,
                    ck,
                    component,
                    np.array(waveform) * coarse_shadow_source_scale,
                )
            ]
        if coarse_shadow_source_projection == "fine_node_nearest":
            ci, cj, ck = grid_coarse.position_to_index(_fine_idx_to_position(fine_idx))
            return [
                (
                    ci,
                    cj,
                    ck,
                    component,
                    np.array(waveform) * coarse_shadow_source_scale,
                )
            ]
        if coarse_shadow_source_projection == "physical_trilinear":
            return _coarse_trilinear_entries(pos, component, waveform)
        if coarse_shadow_source_projection == "fine_node_trilinear":
            return _coarse_trilinear_entries(
                _fine_idx_to_position(fine_idx),
                component,
                waveform,
            )
        raise ValueError(
            "unknown coarse_shadow_source_projection="
            f"{coarse_shadow_source_projection!r}"
        )

    axis_map = {"ex": 0, "ey": 1, "ez": 2}

    def _wire_port_cells_f(pe):
        axis = axis_map[pe.component]
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
        return cells

    def _sparam_probe_idx_f(pe):
        if pe.extent is None:
            return _pos_to_fine_idx(pe.position)
        cells = _wire_port_cells_f(pe)
        return cells[len(cells) // 2]

    for pe in sim._ports:
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
            sources_f.append(
                (
                    i,
                    j,
                    k,
                    pe.component,
                    np.array(waveform) * fine_source_scale,
                )
            )
            if inject_sources_on_coarse_shadow:
                sources_c.extend(
                    _coarse_shadow_source_entries(
                        pe.position,
                        idx,
                        pe.component,
                        waveform,
                    )
                )
            continue

        # Port conductance -> sigma. The general axis-aware form is
        # sigma = G * d_parallel / (d_perp1 * d_perp2); the subgrid fine
        # region is a uniform cubic Grid (dx_f = dx_c / ratio, a single
        # scalar — SBP-SAT subgridding is defined only on uniform grids),
        # so d_parallel / (d_perp1 * d_perp2) reduces exactly to 1 / dx_f.
        # This is the correct cubic-cell result, not a hidden cubic
        # assumption — there is no non-cubic subgrid path to be wrong on.
        if pe.extent is not None:
            # Wire port: compute cells manually
            cells = _wire_port_cells_f(pe)
            n_cells = max(len(cells), 1)
            # Distribute port impedance
            sigma_port_per_cell = n_cells / (pe.impedance * dx_f)
            for cell in cells:
                i, j, k = cell
                mats_f = mats_f._replace(
                    sigma=mats_f.sigma.at[i, j, k].add(sigma_port_per_cell))
                if pec_mask_f is not None:
                    pec_mask_f = pec_mask_f.at[i, j, k].set(False)

            # Precompute Cb-corrected waveforms
            if pe.excite and pe.waveform is not None:
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
            if pec_mask_f is not None:
                pec_mask_f = pec_mask_f.at[i, j, k].set(False)

            if pe.excite and pe.waveform is not None:
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

    ntff_box_f = None
    ntff_data_f = None
    if sim._ntff is not None:
        from rfx.farfield import NTFFBox, init_ntff_data

        corner_lo, corner_hi, ntff_freqs = sim._ntff
        lo_idx = _pos_to_fine_idx(corner_lo)
        hi_idx = _pos_to_fine_idx(corner_hi)
        if any(lo >= hi for lo, hi in zip(lo_idx, hi_idx)):
            raise ValueError(
                "subgrid NTFF box corners must map to a non-empty fine-grid "
                f"box; got lo={lo_idx}, hi={hi_idx}"
            )
        ntff_box_f = NTFFBox(
            i_lo=lo_idx[0],
            i_hi=hi_idx[0],
            j_lo=lo_idx[1],
            j_hi=hi_idx[1],
            k_lo=lo_idx[2],
            k_hi=hi_idx[2],
            freqs=jnp.asarray(ntff_freqs, dtype=jnp.float32),
        )
        ntff_data_f = init_ntff_data(ntff_box_f)

    diagnostic_lumped_sparam_freqs = diagnostic_lumped_sparam_freqs_override
    if diagnostic_lumped_sparam_freqs is None:
        diagnostic_lumped_sparam_freqs = ref.get("diagnostic_lumped_sparam_freqs")
    diagnostic_lumped_sparam_indices_f = []
    diagnostic_lumped_sparam_components = []
    diagnostic_lumped_sparam_impedances = []
    diagnostic_lumped_sparam_cell_counts = []
    diagnostic_lumped_sparam_driven_index = None
    if diagnostic_lumped_sparam_freqs is not None:
        diagnostic_port_entries = [
            pe for pe in sim._ports
            if pe.impedance > 0.0
        ]
        for idx_port, pe in enumerate(diagnostic_port_entries):
            if pe.extent is None:
                cell_count = 1
            else:
                cell_count = max(len(_wire_port_cells_f(pe)), 1)
            diagnostic_lumped_sparam_indices_f.append(_sparam_probe_idx_f(pe))
            diagnostic_lumped_sparam_components.append(pe.component)
            diagnostic_lumped_sparam_impedances.append(float(pe.impedance))
            diagnostic_lumped_sparam_cell_counts.append(int(cell_count))
            if diagnostic_lumped_sparam_driven_index is None and pe.excite:
                diagnostic_lumped_sparam_driven_index = idx_port
        if diagnostic_lumped_sparam_driven_index is None:
            if diagnostic_lumped_sparam_driven_index_override is not None:
                diagnostic_lumped_sparam_driven_index = int(
                    diagnostic_lumped_sparam_driven_index_override
                )
            else:
                diagnostic_lumped_sparam_driven_index = int(
                    ref.get("diagnostic_lumped_sparam_driven_index", 0)
                )
        else:
            diagnostic_lumped_sparam_driven_index = int(
                diagnostic_lumped_sparam_driven_index_override
                if diagnostic_lumped_sparam_driven_index_override is not None
                else ref.get(
                    "diagnostic_lumped_sparam_driven_index",
                    diagnostic_lumped_sparam_driven_index,
                ),
            )

    sg_opts = SubgridRunOptions(
        pec_mask_c=pec_mask_coarse,
        pec_mask_f=pec_mask_f if has_pec_f else None,
        sources_f=sources_f,
        sources_c=sources_c,
        probe_indices_f=probe_indices_f,
        probe_components=probe_components,
        lumped_sparam_indices_f=diagnostic_lumped_sparam_indices_f,
        lumped_sparam_components_f=diagnostic_lumped_sparam_components,
        lumped_sparam_impedances_f=diagnostic_lumped_sparam_impedances,
        lumped_sparam_cell_counts_f=diagnostic_lumped_sparam_cell_counts,
        lumped_sparam_freqs_f=diagnostic_lumped_sparam_freqs,
        ntff_box_f=ntff_box_f,
        ntff_data_f=ntff_data_f,
        # Use the impedance-upwind Maxwell SAT path for all full-x/y z-slab
        # artificial interfaces.  Its vacuum limit is the physically relevant
        # cross-coupled interface condition.  An undocumented refinement-dict
        # override is kept only for research diagnostics that compare the
        # legacy same-kind averaging SAT against the production-candidate path.
        use_material_sat=ref.get("use_material_sat", True),
        sync_coarse_interface_from_fine=ref.get("sync_coarse_interface_from_fine", False),
        sync_coarse_shadow_from_fine=ref.get("sync_coarse_shadow_from_fine", False),
        sync_box_coarse_shadow_from_fine=ref.get("sync_box_coarse_shadow_from_fine", False),
        mask_coarse_shadow_interior=ref.get("mask_coarse_shadow_interior", False),
        use_exterior_z_interfaces=ref.get("use_exterior_z_interfaces", False),
        use_boundary_terminated_exterior_z_interfaces=(
            bool(ref.get("use_boundary_terminated_exterior_z_interfaces", False))
            or auto_boundary_terminated_exterior
        ),
        ghost_exterior_coarse_shadow_from_fine=ref.get(
            "ghost_exterior_coarse_shadow_from_fine",
            False,
        ),
        material_sat_scale=ref.get("material_sat_scale", 1.0),
        material_sat_coarse_scale=ref.get("material_sat_coarse_scale", 1.0),
        material_sat_fine_scale=ref.get("material_sat_fine_scale", 1.0),
        material_sat_e_coarse_scale=ref.get("material_sat_e_coarse_scale", 1.0),
        material_sat_e_fine_scale=ref.get("material_sat_e_fine_scale", 1.0),
        material_sat_h_coarse_scale=ref.get("material_sat_h_coarse_scale", 1.0),
        material_sat_h_fine_scale=ref.get("material_sat_h_fine_scale", 1.0),
        material_sat_zlo_scale=ref.get("material_sat_zlo_scale", 1.0),
        material_sat_zhi_scale=ref.get("material_sat_zhi_scale", 1.0),
        material_sat_e_zlo_scale=ref.get("material_sat_e_zlo_scale", 1.0),
        material_sat_e_zhi_scale=ref.get("material_sat_e_zhi_scale", 1.0),
        material_sat_h_zlo_scale=ref.get("material_sat_h_zlo_scale", 1.0),
        material_sat_h_zhi_scale=ref.get("material_sat_h_zhi_scale", 1.0),
        material_sat_pair_a_zlo_scale=ref.get("material_sat_pair_a_zlo_scale", 1.0),
        material_sat_pair_b_zlo_scale=ref.get("material_sat_pair_b_zlo_scale", 1.0),
        material_sat_zlo_common_trace_projection=ref.get(
            "material_sat_zlo_common_trace_projection",
            "dual",
        ),
        material_sat_zhi_common_trace_projection=ref.get(
            "material_sat_zhi_common_trace_projection",
            "dual",
        ),
        material_sat_normal_e_scale=ref.get("material_sat_normal_e_scale", 0.0),
        material_sat_zhi_coarse_eps_blend=ref.get("material_sat_zhi_coarse_eps_blend", 0.0),
        defer_material_h_sat_until_after_e=ref.get(
            "defer_material_h_sat_until_after_e",
            False,
        ),
        material_sat_face_projection=ref.get("material_sat_face_projection", "node_adjoint"),
        inject_sources_before_e_coupling=ref.get("inject_sources_before_e_coupling", False),
        use_exterior_box_interfaces=ref.get("use_exterior_box_interfaces", False),
        inject_sources_on_coarse_shadow=inject_sources_on_coarse_shadow,
    )
    result = _run_sg(
        grid_coarse,
        base_materials_coarse,
        mats_f,
        config,
        n_steps,
        opts=sg_opts,
    )

    s_params = None
    freqs = None
    if result.lumped_sparam_v_dft_f is not None and result.lumped_sparam_i_dft_f is not None:
        v_dft = result.lumped_sparam_v_dft_f
        i_dft = result.lumped_sparam_i_dft_f
        impedances = result.lumped_sparam_impedances_f
        cell_counts = result.lumped_sparam_cell_counts_f
        freqs = np.asarray(result.lumped_sparam_freqs_f)
        n_ports = int(v_dft.shape[0])
        n_freqs = int(v_dft.shape[1])
        driven = int(diagnostic_lumped_sparam_driven_index or 0)
        if not (0 <= driven < n_ports):
            raise ValueError(
                f"diagnostic_lumped_sparam_driven_index={driven} is outside "
                f"the diagnostic lumped-port count {n_ports}"
            )
        z0_j = impedances[driven]
        n_cells_j = jnp.maximum(cell_counts[driven], 1.0)
        z0_cell_j = z0_j / n_cells_j
        a_j = (-v_dft[driven] + z0_cell_j * i_dft[driven]) / (
            2.0 * jnp.sqrt(z0_cell_j)
        )
        safe_a = jnp.where(jnp.abs(a_j) > 0, a_j, jnp.ones_like(a_j))
        s_mat = jnp.zeros((n_ports, n_ports, n_freqs), dtype=jnp.complex64)
        for recv in range(n_ports):
            z0_i = impedances[recv]
            n_cells_i = jnp.maximum(cell_counts[recv], 1.0)
            if recv == driven and float(n_cells_i) > 1.0:
                safe_i = jnp.where(
                    jnp.abs(i_dft[driven]) > 0,
                    i_dft[driven],
                    jnp.ones_like(i_dft[driven]) * 1e-30,
                )
                z_in = -v_dft[driven] / safe_i
                s_value = (z_in - z0_i) / (z_in + z0_i)
            else:
                z0_cell_i = z0_i / n_cells_i
                b_i = (-v_dft[recv] - z0_cell_i * i_dft[recv]) / (
                    2.0 * jnp.sqrt(z0_cell_i)
                )
                s_value = b_i / safe_a
            s_mat = s_mat.at[recv, driven, :].set(s_value)
        s_params = np.asarray(s_mat)

    return Result(
        state=result.state_f,
        time_series=result.time_series,
        s_params=s_params,
        freqs=freqs,
        ntff_data=result.ntff_data_f,
        ntff_box=result.ntff_box_f,
        grid=fine_grid,
        dt=dt,
        freq_range=(sim._freq_max / 10, sim._freq_max, 'cpml'),
    )


def run_subgridded_path(
    sim,
    grid_coarse,
    base_materials_coarse,
    pec_mask_coarse,
    n_steps,
    *,
    compute_s_params=None,
    s_param_freqs=None,
    s_param_n_steps=None,
):
    """Run simulation using SBP-SAT subgridding (JIT-compiled).

    Inside the guarded one-sided PEC/no-CPML production envelope, explicit
    ``compute_s_params=True`` uses the same V/I replay machinery as the private
    diagnostic to populate a full single-cell lumped-port S-matrix.  Unsupported
    subgrid S-parameter configurations are rejected by the public request and
    validation layers before this runner is reached.
    """
    main_result = _run_subgridded_once(
        sim,
        grid_coarse,
        base_materials_coarse,
        pec_mask_coarse,
        n_steps,
    )

    requested_sparams = (
        compute_s_params is True
        or s_param_freqs is not None
        or s_param_n_steps is not None
    )
    if compute_s_params is None:
        compute_s_params = requested_sparams
    if not compute_s_params:
        return main_result

    port_entries = [
        pe for pe in sim._ports
        if pe.impedance > 0.0
    ]
    if not port_entries:
        raise ValueError(
            "subgrid compute_s_params requires at least one "
            "add_port(..., impedance>0) entry"
        )
    missing_waveform_ports = [
        idx for idx, pe in enumerate(port_entries) if pe.waveform is None
    ]
    if missing_waveform_ports:
        raise ValueError(
            "subgrid compute_s_params needs a waveform on every impedance "
            "port so each port can be driven in turn; missing waveform for "
            f"port index(es) {missing_waveform_ports}"
        )

    freqs = (
        jnp.asarray(s_param_freqs, dtype=jnp.float32)
        if s_param_freqs is not None
        else jnp.linspace(sim._freq_max / 10, sim._freq_max, 50, dtype=jnp.float32)
    )
    sp_n_steps = int(s_param_n_steps if s_param_n_steps is not None else n_steps)
    n_ports = len(port_entries)
    n_freqs = int(freqs.shape[0])
    s_matrix = np.zeros((n_ports, n_ports, n_freqs), dtype=np.complex64)

    original_ports = sim._ports
    try:
        for driven in range(n_ports):
            sim._ports = [
                replace(pe, excite=(idx == driven))
                for idx, pe in enumerate(port_entries)
            ]
            column_result = _run_subgridded_once(
                sim,
                grid_coarse,
                base_materials_coarse,
                pec_mask_coarse,
                sp_n_steps,
                diagnostic_lumped_sparam_freqs_override=freqs,
                diagnostic_lumped_sparam_driven_index_override=driven,
            )
            if column_result.s_params is None:
                raise RuntimeError(
                    "internal subgrid S-parameter replay did not return "
                    f"a column for driven port {driven}"
                )
            s_matrix[:, driven, :] = np.asarray(column_result.s_params)[:, driven, :]
    finally:
        sim._ports = original_ports

    return main_result._replace(
        s_params=s_matrix,
        freqs=np.asarray(freqs),
    )
