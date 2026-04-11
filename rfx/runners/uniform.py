"""Uniform grid run path extracted from Simulation."""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from rfx.simulation import (
    make_source, make_j_source, make_probe, make_port_source, make_wire_port_sources,
    run as _run, run_until_decay as _run_until_decay,
)
from rfx.sources.sources import LumpedPort, setup_lumped_port, WirePort, setup_wire_port
from rfx.probes.probes import extract_s_matrix, extract_s_matrix_wire, init_dft_plane_probe, init_flux_monitor
from rfx.sources.waveguide_port import (
    extract_waveguide_sparams,
    waveguide_plane_positions,
)
from rfx.farfield import make_ntff_box
from rfx.lumped import setup_rlc_materials, build_rlc_meta


def run_uniform(
    sim,
    *,
    n_steps: int,
    until_decay=None,
    decay_check_interval: int = 50,
    decay_min_steps: int = 100,
    decay_max_steps: int = 50_000,
    decay_monitor_component: str = "ez",
    decay_monitor_position=None,
    checkpoint: bool = False,
    compute_s_params=None,
    s_param_freqs=None,
    s_param_n_steps=None,
    snapshot=None,
    subpixel_smoothing: bool = False,
    conformal_pec: bool = False,
    conformal_min_weight: float = 0.1,
    pec_shapes=None,
    # pre-built grid and materials passed in from Simulation.run()
    grid=None,
    base_materials=None,
    debye_spec=None,
    lorentz_spec=None,
    pec_mask=None,
    kerr_chi3=None,
    field_dtype=None,
):
    """Run the uniform-grid simulation path.

    Parameters
    ----------
    sim : Simulation
        The Simulation instance (read-only access to its fields).
    n_steps : int
        Number of timesteps.
    grid : Grid
        Pre-built uniform grid.
    base_materials : MaterialArrays
        Material arrays (before port impedance loading).
    debye_spec, lorentz_spec : dispersion specs or None
    pec_mask : jnp.ndarray or None
    conformal_pec : bool
        Enable Dey-Mittra conformal PEC (default False).
    conformal_min_weight : float
        Minimum conformal weight for CFL stability (default 0.1).
    pec_shapes : list or None
        List of PEC Shape objects (needed for conformal weights).
    All other parameters mirror Simulation.run().

    Returns
    -------
    Result
    """
    from rfx.api import Result, WaveguideSParamResult

    materials = base_materials

    # Fold RLC R and C into material arrays before port/source setup
    for spec in sim._lumped_rlc:
        materials = setup_rlc_materials(grid, spec, materials)

    # Compute per-component smoothed permittivity if requested
    aniso_eps = None
    if subpixel_smoothing:
        from rfx.geometry.smoothing import compute_smoothed_eps
        shape_eps_pairs = [
            (entry.shape, sim._resolve_material(entry.material_name).eps_r)
            for entry in sim._geometry
        ]
        if shape_eps_pairs:
            aniso_eps = compute_smoothed_eps(grid, shape_eps_pairs, background_eps=1.0)

    # Conformal PEC: compute weights and produce anisotropic eps
    conformal_weights = None
    if conformal_pec and pec_shapes:
        from rfx.geometry.conformal import (
            compute_conformal_weights_sdf,
            clamp_conformal_weights,
            conformal_eps_correction,
        )
        w_ex, w_ey, w_ez = compute_conformal_weights_sdf(grid, pec_shapes)
        w_ex, w_ey, w_ez = clamp_conformal_weights(w_ex, w_ey, w_ez, conformal_min_weight)
        conformal_weights = (w_ex, w_ey, w_ez)

        # Compute conformal eps correction
        eps_base = materials.eps_r
        eps_ex_c, eps_ey_c, eps_ez_c = conformal_eps_correction(eps_base, w_ex, w_ey, w_ez)

        if aniso_eps is not None:
            # Merge: conformal PEC overrides smoothed eps at PEC boundary cells
            s_ex, s_ey, s_ez = aniso_eps
            boundary_ex = w_ex < 1.0
            boundary_ey = w_ey < 1.0
            boundary_ez = w_ez < 1.0
            eps_ex_c = jnp.where(boundary_ex, eps_ex_c, s_ex)
            eps_ey_c = jnp.where(boundary_ey, eps_ey_c, s_ey)
            eps_ez_c = jnp.where(boundary_ez, eps_ez_c, s_ez)

        aniso_eps = (eps_ex_c, eps_ey_c, eps_ez_c)

        # Conformal replaces binary pec_mask
        pec_mask = None

    # Build sources and probes for the compiled runner
    sources = []
    probes = []
    dft_planes = []
    waveguide_ports = []
    periodic = None
    cpml_axes = "xyz"
    pec_axes = None
    tfsf = None

    if sim._tfsf is not None and sim._ports:
        raise ValueError(
            "TFSF plane-wave source is not supported together with lumped ports"
        )
    if sim._waveguide_ports and (sim._ports or sim._tfsf):
        raise ValueError(
            "Waveguide ports are not supported together with lumped ports or TFSF"
        )
    if len(sim._waveguide_ports) > 1:
        raise ValueError(
            "Simulation.run() supports only a single waveguide port; use compute_waveguide_s_matrix() for the multiport waveguide scattering workflow"
        )
    if sim._periodic_axes:
        periodic = tuple(axis in sim._periodic_axes for axis in "xyz")

    # Port sources — fold impedances into materials first
    lumped_ports = []
    wire_ports = []
    for pe in sim._ports:
        if pe.impedance == 0.0:
            # Auto-select source type based on boundary conditions:
            # - CPML (open): Cb/dx normalized (prevents DC on PEC surface)
            # - PEC (closed): raw field add (broadband, exact cavity modes)
            if sim._boundary in ("cpml", "upml"):
                sources.append(make_j_source(grid, pe.position, pe.component,
                                             pe.waveform, n_steps, materials))
            else:
                sources.append(make_source(grid, pe.position, pe.component,
                                           pe.waveform, n_steps))
            continue
        if pe.extent is not None:
            # Multi-cell wire port
            axis_map = {"ex": 0, "ey": 1, "ez": 2}
            axis = axis_map[pe.component]
            end = list(pe.position)
            end[axis] += pe.extent
            wp = WirePort(
                start=pe.position, end=tuple(end),
                component=pe.component,
                impedance=pe.impedance, excitation=pe.waveform,
            )
            wire_ports.append(wp)
            materials = setup_wire_port(grid, wp, materials)
            sources.extend(make_wire_port_sources(grid, wp, materials, n_steps))
            # Clear PEC mask at wire cells (probe pierces ground plane)
            if pec_mask is not None:
                from rfx.sources.sources import _wire_port_cells
                for cell in _wire_port_cells(grid, wp):
                    pec_mask = pec_mask.at[cell[0], cell[1], cell[2]].set(False)
        else:
            # Single-cell lumped port
            lp = LumpedPort(
                position=pe.position, component=pe.component,
                impedance=pe.impedance, excitation=pe.waveform,
            )
            lumped_ports.append(lp)
            materials = setup_lumped_port(grid, lp, materials)
            sources.append(make_port_source(grid, lp, materials, n_steps))
            # Clear PEC mask at lumped port cell
            if pec_mask is not None:
                idx = grid.position_to_index(pe.position)
                pec_mask = pec_mask.at[idx[0], idx[1], idx[2]].set(False)

    # Build wire port S-param specs for JIT-integrated DFT
    wire_sparam_specs = []
    if wire_ports and (compute_s_params is None or compute_s_params):
        from rfx.simulation import WirePortSParamSpec
        from rfx.sources.sources import _wire_port_cells
        sp_freqs = s_param_freqs if s_param_freqs is not None else jnp.linspace(
            sim._freq_max / 10, sim._freq_max, 50)
        for wp in wire_ports:
            cells = _wire_port_cells(grid, wp)
            mid = cells[len(cells) // 2]
            wire_sparam_specs.append(WirePortSParamSpec(
                mid_i=mid[0], mid_j=mid[1], mid_k=mid[2],
                component=wp.component,
                freqs=jnp.asarray(sp_freqs, dtype=jnp.float32),
                impedance=wp.impedance,
            ))

    for pe in sim._probes:
        probes.append(make_probe(grid, pe.position, pe.component))

    axis_to_index = {"x": 0, "y": 1, "z": 2}
    for pe in sim._dft_planes:
        axis_idx = axis_to_index[pe.axis]
        plane_pos = [0.0, 0.0, 0.0]
        plane_pos[axis_idx] = pe.coordinate
        grid_index = grid.position_to_index(tuple(plane_pos))[axis_idx]
        freqs_arr = (
            pe.freqs
            if pe.freqs is not None
            else jnp.linspace(sim._freq_max / 10, sim._freq_max, pe.n_freqs)
        )
        dft_planes.append(
            init_dft_plane_probe(
                axis=axis_idx,
                index=grid_index,
                component=pe.component,
                freqs=freqs_arr,
                grid_shape=grid.shape,
                dft_total_steps=n_steps,
            )
        )

    # Flux monitors (Poynting flux through plane)
    flux_monitors = []
    for pe in getattr(sim, '_flux_monitors', []):
        axis_idx = axis_to_index[pe.axis]
        plane_pos = [0.0, 0.0, 0.0]
        plane_pos[axis_idx] = pe.coordinate
        grid_index = grid.position_to_index(tuple(plane_pos))[axis_idx]
        freqs_arr = (
            pe.freqs
            if pe.freqs is not None
            else jnp.linspace(sim._freq_max / 10, sim._freq_max, pe.n_freqs)
        )
        # Compute tangential index bounds from size (finite flux region)
        tangential_axes = [a for a in range(3) if a != axis_idx]
        domain_sizes = [sim._domain[a] for a in tangential_axes]
        grid_ns = [grid.shape[a] for a in tangential_axes]
        if pe.size is not None:
            # User-specified or default center (domain midpoint)
            user_center = pe.center if hasattr(pe, 'center') and pe.center is not None else None
            bounds = []
            for idx_t, (s, dom, n) in enumerate(zip(pe.size, domain_sizes, grid_ns)):
                c = user_center[idx_t] if user_center is not None else dom / 2.0
                pad = getattr(grid, ['pad_x', 'pad_y', 'pad_z'][tangential_axes[idx_t]], 0)
                # Convert physical coordinate to grid index (add CPML padding offset)
                lo = max(0, int(round(c / grid.dx - s / (2.0 * grid.dx))) + pad)
                hi = min(n, int(round(c / grid.dx + s / (2.0 * grid.dx))) + pad)
                bounds.append((lo, hi))
            lo1, hi1 = bounds[0]
            lo2, hi2 = bounds[1]
        else:
            lo1, hi1 = 0, grid_ns[0]
            lo2, hi2 = 0, grid_ns[1]
        flux_monitors.append(
            init_flux_monitor(
                axis=axis_idx,
                index=grid_index,
                freqs=freqs_arr,
                grid_shape=grid.shape,
                dx=grid.dx,
                dft_total_steps=n_steps,
                dft_window=getattr(pe, 'dft_window', 'rect'),
                dft_window_alpha=getattr(pe, 'dft_window_alpha', 0.25),
                lo1=lo1, hi1=hi1, lo2=lo2, hi2=hi2,
            )
        )

    if sim._waveguide_ports:
        cpml_axes = grid.cpml_axes
        pec_axes = "".join(axis for axis in "xyz" if axis not in cpml_axes)
        for pe in sim._waveguide_ports:
            freqs_arr = (
                pe.freqs
                if pe.freqs is not None
                else jnp.linspace(sim._freq_max / 10, sim._freq_max, pe.n_freqs)
            )
            waveguide_ports.append(sim._build_waveguide_port_config(pe, grid, freqs_arr, n_steps))

    if sim._tfsf is not None:
        from rfx.sources.tfsf import init_tfsf

        tfsf = init_tfsf(
            grid.nx,
            grid.dx,
            grid.dt,
            cpml_layers=grid.cpml_layers,
            tfsf_margin=sim._tfsf.margin,
            f0=sim._tfsf.f0 if sim._tfsf.f0 is not None else sim._freq_max / 2,
            bandwidth=sim._tfsf.bandwidth,
            amplitude=sim._tfsf.amplitude,
            polarization=sim._tfsf.polarization,
            direction=sim._tfsf.direction,
            angle_deg=sim._tfsf.angle_deg,
            ny=grid.ny,
            nz=grid.nz,
            waveform=getattr(sim._tfsf, 'waveform', 'differentiated_gaussian'),
        )
        sim._validate_tfsf_vacuum_boundary(materials, tfsf[0])
        periodic = (False, True, True)
        cpml_axes = "x"

    # Floquet port sources — inject plane wave via standard source mechanism
    floquet_port_configs = []
    axis_map_str = {"x": 0, "y": 1, "z": 2}
    for fpe in sim._floquet_ports:
        axis_idx = axis_map_str[fpe.axis]
        pos_vec = [0.0, 0.0, 0.0]
        pos_vec[axis_idx] = fpe.position
        port_grid_index = grid.position_to_index(tuple(pos_vec))[axis_idx]

        fp_f0 = fpe.f0 if fpe.f0 is not None else sim._freq_max / 2
        fp_freqs = (
            fpe.freqs
            if fpe.freqs is not None
            else jnp.linspace(sim._freq_max / 10, sim._freq_max, fpe.n_freqs)
        )

        floquet_port_configs.append({
            "name": fpe.name,
            "axis": axis_idx,
            "port_index": port_grid_index,
            "scan_theta": fpe.scan_theta,
            "scan_phi": fpe.scan_phi,
            "polarization": fpe.polarization,
            "n_modes": fpe.n_modes,
            "freqs": fp_freqs,
            "f0": fp_f0,
            "bandwidth": fpe.bandwidth,
            "amplitude": fpe.amplitude,
        })

        # Add a soft source at the port plane (uniform across the plane)
        from rfx.sources.sources import GaussianPulse as GP
        wf = GP(f0=fp_f0, bandwidth=fpe.bandwidth, amplitude=fpe.amplitude)
        # Place source at center of transverse plane
        center = [sim._domain[i] / 2.0 for i in range(3)]
        center[axis_idx] = fpe.position
        if fpe.polarization == "te":
            # TE: inject first tangential E component
            if fpe.axis == "z":
                comp = "ex"
            elif fpe.axis == "x":
                comp = "ey"
            else:
                comp = "ex"
        else:
            # TM: inject second tangential E component
            if fpe.axis == "z":
                comp = "ey"
            elif fpe.axis == "x":
                comp = "ez"
            else:
                comp = "ez"
        from rfx.simulation import make_source as _make_src
        sources.append(_make_src(grid, tuple(center), comp, wf, n_steps))

    _, debye, lorentz = sim._init_dispersion(
        materials, grid.dt, debye_spec, lorentz_spec)

    # NTFF box
    ntff_box = None
    if sim._ntff is not None:
        corner_lo, corner_hi, freqs = sim._ntff
        ntff_box = make_ntff_box(grid, corner_lo, corner_hi, freqs)

    # Lumped RLC elements
    rlc_metas = None
    if sim._lumped_rlc:
        rlc_metas = [build_rlc_meta(grid, spec, materials) for spec in sim._lumped_rlc]

    # Main simulation
    if until_decay is not None:
        sim_result = _run_until_decay(
            grid, materials,
            decay_by=until_decay,
            check_interval=decay_check_interval,
            min_steps=decay_min_steps,
            max_steps=decay_max_steps,
            monitor_component=decay_monitor_component,
            monitor_position=decay_monitor_position,
            boundary=sim._boundary,
            cpml_axes=cpml_axes,
            pec_axes=pec_axes,
            periodic=periodic,
            debye=debye,
            lorentz=lorentz,
            tfsf=tfsf,
            sources=sources,
            probes=probes,
            dft_planes=dft_planes,
            flux_monitors=flux_monitors,
            waveguide_ports=waveguide_ports,
            ntff=ntff_box,
            snapshot=snapshot,
            checkpoint=checkpoint,
            aniso_eps=aniso_eps,
            pec_mask=pec_mask,
            conformal_weights=conformal_weights,
            wire_port_sparams=wire_sparam_specs or None,
            lumped_rlc=rlc_metas,
            kerr_chi3=kerr_chi3,
            field_dtype=field_dtype,
        )
    else:
        sim_result = _run(
            grid, materials, n_steps,
            boundary=sim._boundary,
            cpml_axes=cpml_axes,
            pec_axes=pec_axes,
            periodic=periodic,
            debye=debye,
            lorentz=lorentz,
            tfsf=tfsf,
            sources=sources,
            probes=probes,
            dft_planes=dft_planes,
            flux_monitors=flux_monitors,
            waveguide_ports=waveguide_ports,
            ntff=ntff_box,
            snapshot=snapshot,
            checkpoint=checkpoint,
            aniso_eps=aniso_eps,
            pec_mask=pec_mask,
            conformal_weights=conformal_weights,
            wire_port_sparams=wire_sparam_specs or None,
            lumped_rlc=rlc_metas,
            kerr_chi3=kerr_chi3,
            field_dtype=field_dtype,
        )

    # S-parameters: use JIT-integrated DFT for wire ports (fast),
    # fall back to Python loop for lumped ports
    if compute_s_params is None:
        compute_s_params = len(lumped_ports) > 0 or len(wire_ports) > 0

    s_params = None
    freqs_out = None

    if compute_s_params and wire_ports and sim_result.wire_port_sparams:
        # Extract S-params from JIT-accumulated DFTs (100x faster)
        # NOTE: The JIT scan body accumulates V/I after source injection.
        # For accurate S-params the Python-loop path
        # (extract_s_matrix_wire) should be preferred; this fast path
        # gives a usable but less accurate approximation.
        n_wp = len(wire_ports)
        if s_param_freqs is None:
            s_param_freqs = wire_ports[0].excitation.f0  # placeholder
            # Use freqs from the first wire port spec
            for wp_meta, accs in sim_result.wire_port_sparams:
                s_param_freqs = np.array(wp_meta.freqs)
                break
        freqs_out = np.array(s_param_freqs)
        n_freqs = len(freqs_out)
        S = np.zeros((n_wp, n_wp, n_freqs), dtype=np.complex64)

        for j, (wp_meta, accs) in enumerate(sim_result.wire_port_sparams):
            v_dft, i_dft, _ = accs
            z0 = wp_meta.impedance
            # Wave decomposition with FDTD sign convention (V = -E·dx)
            a_j = (-v_dft + z0 * i_dft) / (2.0 * np.sqrt(z0))
            safe_a = jnp.where(jnp.abs(a_j) > 0, a_j, jnp.ones_like(a_j))
            b_j = (-v_dft - z0 * i_dft) / (2.0 * np.sqrt(z0))
            S[j, j, :] = np.array(b_j / safe_a)

        s_params = S
    elif compute_s_params and (lumped_ports or wire_ports):
        # Fall back to Python-loop extraction
        if s_param_freqs is None:
            s_param_freqs = jnp.linspace(
                sim._freq_max / 10, sim._freq_max, 50,
            )
        freqs_out = np.array(s_param_freqs)

        # Default to the main simulation's n_steps so the S-param
        # extraction runs long enough for high-Q cavities to ring down.
        sp_n_steps = s_param_n_steps if s_param_n_steps is not None else n_steps

        if wire_ports:
            s_params = extract_s_matrix_wire(
                grid, base_materials, wire_ports, s_param_freqs,
                n_steps=sp_n_steps,
                boundary=sim._boundary,
                debye_spec=debye_spec,
                lorentz_spec=lorentz_spec,
                pec_mask=pec_mask,
            )
        else:
            s_params = extract_s_matrix(
                grid, base_materials, lumped_ports, s_param_freqs,
                n_steps=sp_n_steps,
                boundary=sim._boundary,
                debye_spec=debye_spec,
                lorentz_spec=lorentz_spec,
            )

    waveguide_ports_result = (
        {
            entry.name: cfg
            for entry, cfg in zip(sim._waveguide_ports, sim_result.waveguide_ports or ())
        }
        if sim._waveguide_ports
        else None
    )
    waveguide_sparams_result = None
    if sim._waveguide_ports:
        waveguide_sparams_result = {}
        for entry, cfg in zip(sim._waveguide_ports, sim_result.waveguide_ports or ()):
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

    return Result(
        state=sim_result.state,
        time_series=sim_result.time_series,
        s_params=s_params,
        freqs=freqs_out,
        ntff_data=sim_result.ntff_data,
        ntff_box=ntff_box,
        dft_planes=(
            {
                entry.name: probe
                for entry, probe in zip(sim._dft_planes, sim_result.dft_planes or ())
            }
            if sim._dft_planes
            else None
        ),
        flux_monitors=(
            {
                entry.name: fm
                for entry, fm in zip(
                    getattr(sim, '_flux_monitors', []),
                    sim_result.flux_monitors or (),
                )
            }
            if getattr(sim, '_flux_monitors', [])
            else None
        ),
        waveguide_ports=waveguide_ports_result,
        waveguide_sparams=waveguide_sparams_result,
        snapshots=sim_result.snapshots,
        grid=grid,
        dt=grid.dt,
        freq_range=(sim._freq_max / 10, sim._freq_max, sim._boundary),
    )
