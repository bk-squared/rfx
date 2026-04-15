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
    *,
    dx_profile: np.ndarray | None = None,
    dy_profile: np.ndarray | None = None,
) -> NonUniformGrid:
    """Build a NonUniformGrid from per-axis profiles.

    ``dx_profile`` / ``dy_profile`` are optional; when omitted the
    corresponding axis is uniform with spacing ``dx`` across
    ``domain``.
    """
    if dx is None:
        dx = C0 / freq_max / 20.0
    domain_xy = (domain[0], domain[1])
    return make_nonuniform_grid(
        domain_xy, dz_profile, dx, cpml_layers,
        dx_profile=dx_profile, dy_profile=dy_profile,
    )


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
    """Convert physical (x, y, z) to non-uniform grid indices.

    Delegates to ``rfx.nonuniform.position_to_index`` so the cumulative
    lookup works on non-uniform xy as well.
    """
    from rfx.nonuniform import position_to_index
    return position_to_index(grid, pos)


def _build_waveguide_port_config_nu(sim, entry, grid: NonUniformGrid,
                                     freqs: jnp.ndarray, n_steps: int):
    """NU-aware waveguide port config builder.

    Mirrors the uniform-path ``Simulation._build_waveguide_port_config``
    but resolves indices and per-axis aperture spans against the
    NonUniformGrid (cumulative-sum cell edges).
    """
    from rfx.sources.waveguide_port import (
        WaveguidePort,
        init_waveguide_port,
        init_multimode_waveguide_port,
    )

    normal_axis = entry.direction[1]
    axis_idx = {"x": 0, "y": 1, "z": 2}[normal_axis]
    pos_vec = [0.0, 0.0, 0.0]
    pos_vec[axis_idx] = entry.x_position
    x_index = pos_to_nu_index(grid, tuple(pos_vec))[axis_idx]

    cpml = grid.cpml_layers
    # axis-pad mirrors Grid.axis_pads convention: 0 if no CPML on that axis,
    # else cpml. NU runner uses CPML on all axes when boundary='cpml'/'upml'.
    axis_pads_nu = (cpml, cpml, cpml)

    def _range_to_slice_nu(value_range, d_arr_jnp, n_axis, pad):
        d_np = np.asarray(d_arr_jnp)
        # Cell-edge positions in physical coords (interior only, edge=0 at first
        # interior face). Length = n_interior + 1.
        interior = d_np[pad : n_axis - pad]
        edges = np.insert(np.cumsum(interior), 0, 0.0)
        if value_range is None:
            return (pad, n_axis - pad), float(edges[-1])
        lo, hi = value_range
        lo_local = int(np.argmin(np.abs(edges - float(lo))))
        hi_local = int(np.argmin(np.abs(edges - float(hi))))
        if hi_local <= lo_local:
            raise ValueError(
                f"range {value_range!r} does not resolve to a valid aperture on the NU grid"
            )
        lo_idx = lo_local + pad
        hi_idx = hi_local + pad + 1
        actual_span = float(edges[hi_local] - edges[lo_local])
        if actual_span <= 0.0:
            raise ValueError(
                f"range {value_range!r} resolves to invalid aperture span {actual_span}"
            )
        return (lo_idx, hi_idx), actual_span

    if normal_axis == "x":
        u_slice, a_span = _range_to_slice_nu(entry.y_range, grid.dy_arr, grid.ny, axis_pads_nu[1])
        v_slice, b_span = _range_to_slice_nu(entry.z_range, grid.dz, grid.nz, axis_pads_nu[2])
    elif normal_axis == "y":
        u_slice, a_span = _range_to_slice_nu(entry.x_range, grid.dx_arr, grid.nx, axis_pads_nu[0])
        v_slice, b_span = _range_to_slice_nu(entry.z_range, grid.dz, grid.nz, axis_pads_nu[2])
    else:
        u_slice, a_span = _range_to_slice_nu(entry.x_range, grid.dx_arr, grid.nx, axis_pads_nu[0])
        v_slice, b_span = _range_to_slice_nu(entry.y_range, grid.dy_arr, grid.ny, axis_pads_nu[1])

    # Snapped source-plane physical coordinate (for waveguide_plane_positions).
    # Use the cumulative cell-edge position corresponding to the snapped cell
    # index along the port-normal axis.
    if normal_axis == "x":
        d_axis_np = np.asarray(grid.dx_arr)
    elif normal_axis == "y":
        d_axis_np = np.asarray(grid.dy_arr)
    else:
        d_axis_np = np.asarray(grid.dz)
    _interior = d_axis_np[cpml : len(d_axis_np) - cpml]
    _edges_axis = np.insert(np.cumsum(_interior), 0, 0.0)
    _local_axis = max(0, min(x_index - cpml, len(_edges_axis) - 1))
    snapped_source_plane = float(_edges_axis[_local_axis])

    port = WaveguidePort(
        x_index=x_index,
        y_slice=None,
        z_slice=None,
        a=a_span,
        b=b_span,
        mode=entry.mode,
        mode_type=entry.mode_type,
        direction=entry.direction,
        x_position=snapped_source_plane,
        normal_axis=normal_axis,
        u_slice=u_slice,
        v_slice=v_slice,
    )
    if entry.n_modes > 1:
        return init_multimode_waveguide_port(
            port, grid, freqs,
            n_modes=entry.n_modes,
            f0=entry.f0 if entry.f0 is not None else sim._freq_max / 2,
            bandwidth=entry.bandwidth,
            amplitude=entry.amplitude,
            probe_offset=entry.probe_offset,
            ref_offset=entry.ref_offset,
            dft_total_steps=n_steps,
        )
    return init_waveguide_port(
        port, grid, freqs,
        f0=entry.f0 if entry.f0 is not None else sim._freq_max / 2,
        bandwidth=entry.bandwidth,
        amplitude=entry.amplitude,
        probe_offset=entry.probe_offset,
        ref_offset=entry.ref_offset,
        dft_total_steps=n_steps,
    )


def run_nonuniform_path(sim, *, n_steps, compute_s_params=None, s_param_freqs=None,
                        eps_override=None, sigma_override=None,
                        pec_mask_override=None):
    """Run simulation on non-uniform grid with graded dz.

    Parameters
    ----------
    sim : Simulation
        The Simulation instance (read-only access to its fields).
    n_steps : int
        Number of timesteps.
    compute_s_params : bool or None
    s_param_freqs : array or None
    eps_override, sigma_override : jnp.ndarray or None
        When provided, replace the assembled material arrays before
        source/port setup. Used by the differentiable ``forward()``
        path to inject optimisation variables.
    pec_mask_override : jnp.ndarray or None
        Extra hard-PEC mask ORed into the geometry-derived pec_mask.

    Returns
    -------
    Result
    """
    from rfx.api import Result

    grid = build_nonuniform_grid(
        sim._freq_max, sim._domain, sim._dx, sim._cpml_layers, sim._dz_profile,
        dx_profile=sim._dx_profile, dy_profile=sim._dy_profile,
    )
    materials, debye_spec, lorentz_spec, pec_mask = assemble_materials_nu(sim, grid)

    # ``eps_override`` / ``sigma_override`` replace the assembled material
    # arrays for the scan. We keep the original concrete ``materials`` for
    # ``make_current_source`` (which calls ``float(materials.eps_r[i,j,k])``)
    # so that source normalisation stays concrete under ``jax.grad``; the
    # override is folded into a separate ``materials_scan`` used for
    # port-sigma updates and the scan launch.
    materials_concrete = materials
    if eps_override is not None or sigma_override is not None:
        materials = MaterialArrays(
            eps_r=eps_override if eps_override is not None else materials.eps_r,
            sigma=sigma_override if sigma_override is not None else materials.sigma,
            mu_r=materials.mu_r,
        )

    if pec_mask_override is not None:
        pec_mask = pec_mask_override if pec_mask is None else (pec_mask | pec_mask_override)

    # Fold RLC R/C into materials before other port/source setup
    # (mirrors the uniform path).
    if sim._lumped_rlc:
        from rfx.lumped import setup_rlc_materials
        for spec in sim._lumped_rlc:
            materials = setup_rlc_materials(grid, spec, materials)

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

    # Domain extents (for auto-detecting port direction).
    dom_lx = float(sim._domain[0])
    dom_ly = float(sim._domain[1])

    def _auto_direction(position) -> str:
        """Pick the outward-normal direction (+x/-x/+y/-y) of the port
        by finding the smallest *relative* distance to a boundary
        face. Relative (not absolute) distance avoids the trap where
        a short domain extent makes a completely-centered port appear
        "close" to its narrow-axis boundary.
        """
        x, y, _z = position
        rel = {
            "-x": float(x) / max(dom_lx, 1e-30),
            "+x": float(dom_lx - x) / max(dom_lx, 1e-30),
            "-y": float(y) / max(dom_ly, 1e-30),
            "+y": float(dom_ly - y) / max(dom_ly, 1e-30),
        }
        return min(rel, key=rel.get)

    for pe in sim._ports:
        idx = pos_to_nu_index(grid, pe.position)
        if pe.impedance == 0.0:
            # Current source with dV normalization
            src = make_current_source(
                grid, idx, pe.component, pe.waveform, n_steps, materials_concrete)
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

            _dx_np = np.asarray(grid.dx_arr)
            _dy_np = np.asarray(grid.dy_arr)
            for k in wire_cells:
                cell = list(idx)
                cell[axis] = k
                ci, cj, ck = cell
                dxi = float(_dx_np[ci])
                dyj = float(_dy_np[cj])
                # 3D wire port: σ = n_cells * d_parallel / (Z0 * d_perp1 * d_perp2)
                # Each cell in the wire carries 1/n_cells of total impedance Z0.
                if axis == 2:
                    d_cell = float(grid.dz[ck])
                    dp1, dp2 = dxi, dyj
                elif axis == 1:
                    d_cell = dyj
                    dp1, dp2 = dxi, float(grid.dz[ck])
                else:
                    d_cell = dxi
                    dp1, dp2 = dyj, float(grid.dz[ck])
                sigma_port = n_cells * d_cell / (pe.impedance * dp1 * dp2)
                materials = materials._replace(
                    sigma=materials.sigma.at[ci, cj, ck].add(
                        sigma_port))
                if pec_mask is not None:
                    pec_mask = pec_mask.at[ci, cj, ck].set(False)

            # Create per-cell sources — only when the port is excited.
            # Passive (excite=False) ports contribute just the σ
            # resistive termination above, acting as a matched load.
            mid_k = wire_cells[len(wire_cells) // 2]
            mid_cell = list(idx)
            mid_cell[axis] = mid_k

            if pe.excite:
                for k in wire_cells:
                    cell = list(idx)
                    cell[axis] = k
                    src = make_current_source(
                        grid, tuple(cell), pe.component,
                        pe.waveform, n_steps, materials_concrete)
                    # Scale by 1/n_cells for distributed excitation
                    scaled_wf = np.array(src[4]) / n_cells
                    sources.append(
                        (src[0], src[1], src[2], src[3], scaled_wf))

            # Wire port S-param spec — include excite/direction so the
            # runner scan body can record V/I and the post-processing
            # can orient the wave decomposition correctly.
            port_direction = pe.direction or _auto_direction(pe.position)
            wire_port_specs.append({
                'mid_i': mid_cell[0], 'mid_j': mid_cell[1],
                'mid_k': mid_cell[2],
                'component': pe.component,
                'impedance': pe.impedance,
                'excite': bool(pe.excite),
                'direction': port_direction,
            })
        else:
            # Single-cell lumped port
            i, j, k = idx
            dxi = float(np.asarray(grid.dx_arr)[i])
            dyj = float(np.asarray(grid.dy_arr)[j])
            # 3D lumped port: σ = d_parallel / (Z0 * d_perp1 * d_perp2)
            # This ensures correct power dissipation P = V²/Z0 in
            # anisotropic cells where dz ≠ dx.  The old formula
            # σ = 1/(Z0*d_parallel) is only valid for cubic cells.
            axis_map = {"ex": 0, "ey": 1, "ez": 2}
            port_axis = axis_map[pe.component]
            if port_axis == 2:
                d_parallel = float(grid.dz[k])
                d_perp1, d_perp2 = dxi, dyj
            elif port_axis == 1:
                d_parallel = dyj
                d_perp1, d_perp2 = dxi, float(grid.dz[k])
            else:
                d_parallel = dxi
                d_perp1, d_perp2 = dyj, float(grid.dz[k])
            sigma_port = d_parallel / (pe.impedance * d_perp1 * d_perp2)
            materials = materials._replace(
                sigma=materials.sigma.at[i, j, k].add(sigma_port))
            if pec_mask is not None:
                pec_mask = pec_mask.at[i, j, k].set(False)
            if pe.excite:
                src = make_current_source(
                    grid, idx, pe.component, pe.waveform, n_steps, materials_concrete)
                sources.append(src)

    for pe in sim._probes:
        idx = pos_to_nu_index(grid, pe.position)
        probes.append((*idx, pe.component))

    # DFT plane probes — mirror the uniform-path setup in
    # runners/uniform.py, but use pos_to_nu_index so the plane
    # coordinate resolves on a possibly-graded mesh.
    dft_plane_probes = []
    if sim._dft_planes:
        from rfx.probes.probes import init_dft_plane_probe
        axis_to_index = {"x": 0, "y": 1, "z": 2}
        for pe in sim._dft_planes:
            axis_idx = axis_to_index[pe.axis]
            plane_pos = [0.0, 0.0, 0.0]
            plane_pos[axis_idx] = pe.coordinate
            grid_index = pos_to_nu_index(grid, tuple(plane_pos))[axis_idx]
            freqs_arr = (
                pe.freqs
                if pe.freqs is not None
                else jnp.linspace(sim._freq_max / 10, sim._freq_max, pe.n_freqs)
            )
            dft_plane_probes.append(
                init_dft_plane_probe(
                    axis=axis_idx,
                    index=grid_index,
                    component=pe.component,
                    freqs=freqs_arr,
                    grid_shape=(grid.nx, grid.ny, grid.nz),
                    dft_total_steps=n_steps,
                )
            )

    sp_freqs = None
    if wire_port_specs and (compute_s_params is None or compute_s_params):
        sp_freqs = s_param_freqs
        if sp_freqs is None:
            sp_freqs = np.linspace(
                sim._freq_max / 10, sim._freq_max, 50)

    # Lumped RLC: build per-element metadata + zero-init ADE states
    rlc_metas: tuple = ()
    rlc_states_init: tuple = ()
    if sim._lumped_rlc:
        from rfx.lumped import build_rlc_meta, init_rlc_state
        rlc_metas = tuple(
            build_rlc_meta(grid, spec, materials) for spec in sim._lumped_rlc
        )
        rlc_states_init = tuple(init_rlc_state() for _ in sim._lumped_rlc)

    # Waveguide ports: build per-port config via NU-aware
    # init_waveguide_port (duck-types on grid for per-axis widths).
    waveguide_port_cfgs = []
    if sim._waveguide_ports:
        wg_freqs = None
        for pe in sim._waveguide_ports:
            if pe.freqs is not None:
                wg_freqs = jnp.asarray(pe.freqs, dtype=jnp.float32)
                break
        if wg_freqs is None:
            wg_freqs = jnp.linspace(
                sim._freq_max * 0.5, sim._freq_max, 20, dtype=jnp.float32
            )
        for pe in sim._waveguide_ports:
            waveguide_port_cfgs.append(
                _build_waveguide_port_config_nu(
                    sim, pe, grid, wg_freqs, n_steps,
                )
            )

    # TFSF plane-wave source. Scope: axis-aligned +x / -x incidence with
    # angle_deg=0 so the 1D auxiliary grid runs along the uniform x axis
    # with scalar cell size grid.dx. Oblique angles (2D auxiliary grid)
    # and +z / -z incidence (which would require a z-nonuniform 1D aux)
    # are rejected here with actionable messages.
    tfsf_pair = None
    if sim._tfsf is not None:
        entry = sim._tfsf
        if abs(entry.angle_deg) > 0.01:
            raise ValueError(
                "TFSF oblique incidence (angle_deg != 0) is not yet "
                "supported on nonuniform z mesh — the 2D auxiliary grid "
                "is uniform-only. Use angle_deg=0 or run on the uniform lane."
            )
        if entry.direction in ("+z", "-z"):
            raise ValueError(
                "TFSF z-directed incidence is not yet supported on "
                "nonuniform z mesh (1D auxiliary grid would need to be "
                "z-nonuniform). Use direction='+x' or '-x', or run on the "
                "uniform lane."
            )
        if entry.direction not in ("+x", "-x"):
            raise ValueError(
                "TFSF on nonuniform mesh supports only direction='+x' or "
                f"'-x'; got {entry.direction!r}."
            )
        from rfx.sources.tfsf import init_tfsf
        tfsf_pair = init_tfsf(
            grid.nx,
            grid.dx,
            grid.dt,
            cpml_layers=grid.cpml_layers,
            tfsf_margin=entry.margin,
            f0=entry.f0 if entry.f0 is not None else sim._freq_max / 2,
            bandwidth=entry.bandwidth,
            amplitude=entry.amplitude,
            polarization=entry.polarization,
            direction=entry.direction,
            angle_deg=entry.angle_deg,
            ny=grid.ny,
            nz=grid.nz,
            waveform=getattr(entry, 'waveform', 'differentiated_gaussian'),
        )

    # NTFF box: build once (indices are Python-static) + zero-init
    # DFT accumulators that will be threaded through the scan carry.
    # NonUniformGrid is a NamedTuple without ``position_to_index``, so
    # we build the box directly via ``pos_to_nu_index`` (which uses
    # the cumulative dx_arr/dy_arr/dz lookup) and skip make_ntff_box.
    ntff_box = None
    ntff_data_init = None
    if sim._ntff is not None:
        from rfx.farfield import NTFFBox, init_ntff_data
        corner_lo, corner_hi, ntff_freqs = sim._ntff
        lo_idx = pos_to_nu_index(grid, corner_lo)
        hi_idx = pos_to_nu_index(grid, corner_hi)
        ntff_box = NTFFBox(
            i_lo=lo_idx[0], i_hi=hi_idx[0],
            j_lo=lo_idx[1], j_hi=hi_idx[1],
            k_lo=lo_idx[2], k_hi=hi_idx[2],
            freqs=jnp.asarray(ntff_freqs, dtype=jnp.float32),
        )
        ntff_data_init = init_ntff_data(ntff_box)

    r = run_nonuniform(
        grid, materials, n_steps,
        pec_mask=pec_mask,
        sources=sources,
        probes=probes,
        wire_ports=wire_port_specs if wire_port_specs else None,
        s_param_freqs=sp_freqs,
        debye=debye,
        lorentz=lorentz,
        pec_faces=getattr(sim, '_pec_faces', None),
        dft_planes=dft_plane_probes if dft_plane_probes else None,
        rlc_metas=rlc_metas,
        rlc_states=rlc_states_init,
        ntff_box=ntff_box,
        ntff_data=ntff_data_init,
        waveguide_ports=waveguide_port_cfgs if waveguide_port_cfgs else None,
        tfsf=tfsf_pair,
    )

    s_params = r.get("s_params")
    freqs_out = r.get("s_param_freqs")

    # Waveguide-port output: dict[name -> cfg], optional sparams dict.
    waveguide_ports_result = None
    waveguide_sparams_result = None
    if sim._waveguide_ports and "waveguide_ports" in r:
        from rfx.sources.waveguide_port import (
            extract_waveguide_sparams,
            waveguide_plane_positions,
        )
        from rfx.api import WaveguideSParamResult
        final_cfgs = r["waveguide_ports"]
        waveguide_ports_result = {
            entry.name: cfg
            for entry, cfg in zip(sim._waveguide_ports, final_cfgs)
        }
        waveguide_sparams_result = {}
        for entry, cfg in zip(sim._waveguide_ports, final_cfgs):
            plane_positions = waveguide_plane_positions(cfg)
            source_plane = plane_positions["source"]
            measured_reference_plane = plane_positions["reference"]
            measured_probe_plane = plane_positions["probe"]
            if entry.calibration_preset == "source_to_probe":
                reference_plane = source_plane
                probe_plane = measured_probe_plane
                calibration_preset = "source_to_probe"
            elif entry.reference_plane is not None or entry.probe_plane is not None:
                reference_plane = (
                    entry.reference_plane
                    if entry.reference_plane is not None
                    else measured_reference_plane
                )
                probe_plane = (
                    entry.probe_plane
                    if entry.probe_plane is not None
                    else measured_probe_plane
                )
                calibration_preset = "explicit"
            else:
                reference_plane = measured_reference_plane
                probe_plane = measured_probe_plane
                calibration_preset = "measured"
            s11, s21 = extract_waveguide_sparams(
                cfg,
                ref_shift=reference_plane - measured_reference_plane,
                probe_shift=probe_plane - measured_probe_plane,
            )
            waveguide_sparams_result[entry.name] = WaveguideSParamResult(
                freqs=np.array(cfg.freqs),
                s11=np.array(s11),
                s21=np.array(s21),
                calibration_preset=calibration_preset,
                source_plane=float(source_plane),
                measured_reference_plane=measured_reference_plane,
                measured_probe_plane=measured_probe_plane,
                reference_plane=reference_plane,
                probe_plane=probe_plane,
            )

    # Repack DFT planes into {name: DFTPlaneProbe} dict to match
    # the uniform path's Result schema.
    dft_planes_dict = None
    if sim._dft_planes and "dft_planes" in r:
        dft_planes_dict = {
            entry.name: probe
            for entry, probe in zip(sim._dft_planes, r["dft_planes"])
        }

    return Result(
        state=r["state"],
        time_series=r["time_series"],
        s_params=s_params,
        freqs=freqs_out,
        ntff_data=r.get("ntff_data"),
        ntff_box=ntff_box,
        dft_planes=dft_planes_dict,
        waveguide_ports=waveguide_ports_result,
        waveguide_sparams=waveguide_sparams_result,
        grid=grid,
        dt=grid.dt,
        freq_range=(sim._freq_max / 10, sim._freq_max, sim._boundary),
    )
