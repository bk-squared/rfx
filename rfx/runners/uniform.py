"""Uniform grid run path extracted from Simulation."""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from rfx import simulation as _simulation
# Every rfx.simulation symbol this module needs goes through `_simulation.*`
# (never `from rfx.simulation import X as _x`, #628): this module is first
# imported lazily, from inside Simulation.run()/forward(), so an import-time
# alias can capture a test-monkeypatched rfx.simulation attribute
# permanently, even after the patch is torn down — a module reference is
# resolved at CALL time and always reflects the current attribute. These are
# setup-time calls (once per port/source registration), not per-timestep.
from rfx.sources.sources import LumpedPort, setup_lumped_port, WirePort, setup_wire_port
from rfx.probes.probes import init_dft_plane_probe
from rfx.measurement.flux_weights import build_flux_monitor_cfgs
from rfx.measurement import package_declared_measurements
from rfx.current_moments import monitor_for_simulation as _cm_for_sim
from rfx.farfield import make_ntff_box
from rfx.lumped import setup_rlc_materials, build_rlc_meta
from rfx.measurement.oblique import reconstruct_oblique_physical
from rfx.measurement.decay import drive_steps, refuse_radiated_flux_stop



def run_uniform(
    sim,
    *,
    diagnostics=(),
    n_steps: int,
    until_decay=None,
    decay_check_interval: int = 50,
    decay_min_steps: int = 100,
    decay_max_steps: int = 50_000,
    decay_energy_consecutive: int = 2,
    radiated_flux_box: tuple | None = None,
    flux_env_checks: int = 4,
    decay_monitor_component: str = "ez",
    decay_monitor_position=None,
    checkpoint: bool = False,
    compute_s_params=None,
    s_param_freqs=None,
    s_param_n_steps=None,
    snapshot=None,
    subpixel_smoothing: bool | str = False,
    conformal_pec: bool | None = None,
    conformal_min_weight: float = 0.1,
    pec_shapes=None,
    # pre-built grid and materials passed in from Simulation.run()
    grid=None,
    base_materials=None,
    conductors=None,
    debye_spec=None,
    lorentz_spec=None,
    pec_mask=None,
    kerr_chi3=None,
    field_dtype=None,
    report_every: int | None = None,
    report_label: str = "",
    sheet_specs=None,
    pec_sheets=None,
    pec_wires=None,
    keep_wire_port_sparams: bool = True,
    stop_fn=None,
    stop_interval: int = 250,
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
        Primal-cell occupancy of the PEC VOLUMES (#931 §1.1).
    pec_sheets, pec_wires : list or None
        PEC sheets (#931 §1.3) and sub-cell wires (§1.4) collected by the
        assembler; they own no cell, so they reach the stepper only
        through the realized edge masks built here.
    conformal_pec : bool or None
        Enable conformal PEC (eps/w on cut E edges, #1651). None reads
        Boundary(conformal=True); an explicit False requests staircase PEC.
    conformal_min_weight : float
        Minimum cut-edge weight; caps its permittivity at eps/conformal_min_weight (default 0.1).
    pec_shapes : list or None
        List of PEC Shape objects (needed for conformal weights).
    keep_wire_port_sparams : bool
        Hand back the scan's per-wire-port ``(spec, accumulators)`` pairs as
        ``Result.wire_port_sparams`` (default ``True``). Internal callers
        can request ``False``. ``run(..., ringdown=...)`` reads the realized port cells and
        accumulators from them (issue #1254) and preserves them in Result.
    stop_fn, stop_interval
        ``run(..., ringdown=..., until_identified=True)`` (issue #1254): the
        fixed-length scan runs in chunks of ``stop_interval`` steps and
        ``stop_fn(steps_done, peek)`` decides after each one whether the run
        ends there; ``peek()`` gives the record so far (``time_series``,
        ``wire_port_sparams``, ``grid``, ``dt`` and the ``freqs`` this
        function reports). ``None`` (default) is the unchanged path.
    All other parameters mirror Simulation.run().

    Returns
    -------
    Result
    """
    from rfx.api import Result

    refuse_radiated_flux_stop(radiated_flux_box, until_decay, sim._boundary)

    # Step-count-sized builds (drive tables, DFT and flux accumulators, port
    # records): on the decay path the loop runs up to decay_max_steps, so they
    # are sized for it, as the graded runner's sizing_n. Sized by n_steps, a
    # drive still on at n_steps was cut there while the run went on.
    sizing_n = int(decay_max_steps) if until_decay is not None else n_steps
    if until_decay is not None:
        # The decay loop accumulates flux with rectangular weights only.
        from rfx.measurement.decay import refuse_windowed_flux
        refuse_windowed_flux(sim, "uniform")

    if conformal_pec is None:
        conformal_pec = bool(sim._boundary_spec.conformal_faces())

    # run() sends a refined model to the subgridded lane; a direct call must
    # not solve it here without the refinement (#1240).
    sim._require_no_refinement_without_a_subgrid("run_uniform()")

    # #677 v1 fences for the surface-impedance sheet operator (loud, never
    # silent): the operator replaces the standard isotropic E update at its
    # edges, so lanes that swap that update out are refused.
    if sheet_specs:
        if sim._boundary == "upml":
            raise ValueError(
                "surface-impedance (surface_impedance_f0) sheets are not "
                "supported with boundary='upml': UPML replaces "
                "the whole E update with its split-field form, which the "
                "sheet operator would silently override at its edges. Use "
                "boundary='cpml' or 'pec'.")
        if subpixel_smoothing or conformal_pec:
            raise ValueError(
                "surface-impedance (surface_impedance_f0) sheets are not "
                "supported with subpixel_smoothing / conformal_pec"
                ": the sheet operator assumes the plain isotropic "
                "E update at its edges (Stage-1/2 conformal and Kottke "
                "lanes do not apply the sheet operator).")
        if debye_spec is not None or lorentz_spec is not None:
            raise ValueError(
                "surface-impedance (surface_impedance_f0) sheets combined "
                "with dispersive (Debye/Lorentz) materials in one run are "
                "not supported: the sheet operator would "
                "silently override the ADE dispersion update at its "
                "edges. Remove the dispersive material or the f0 sheet.")

    from rfx.model.tensor_operands import refuse_tensor_conflicts
    refuse_tensor_conflicts(sim, grid, subpixel_smoothing=subpixel_smoothing,
        conformal_pec=conformal_pec, debye_spec=debye_spec,
        lorentz_spec=lorentz_spec, pec_shapes=pec_shapes, kerr_chi3=kerr_chi3)

    materials = base_materials

    # Fold RLC R and C into material arrays before port/source setup
    for spec in sim._lumped_rlc:
        materials = setup_rlc_materials(grid, spec, materials)

    use_kottke_pec = (subpixel_smoothing == "kottke_pec")
    from rfx.model.tensor_operands import form_tensor_operands, finish_tensor_operands
    aniso_eps, aniso_inv_eps, conformal_weights, _unextendable = form_tensor_operands(
        sim, grid, materials, subpixel_smoothing=subpixel_smoothing,
        conformal=conformal_pec, conformal_min_weight=conformal_min_weight,
        pec_shapes=pec_shapes)
    if subpixel_smoothing:
        from rfx.geometry.smoothing import warn_unextendable_shapes
        # Assembly already emitted the conductor findings for this run.
        warn_unextendable_shapes([u for u in _unextendable if not u.conductor])
    if sim._waveguide_ports and (conformal_weights is not None
                                 or sim._boundary_spec.conformal_faces()):
        # Dispatch judged the default conformal routing; judge the weights formed here.
        from rfx.boundaries.waveguide_edges import refuse_bare_waveguide_edges
        refuse_bare_waveguide_edges(sim, grid, lane="run_uniform",
                                    conformal_weights=conformal_weights)
    if conformal_weights is not None:
        # Conformal replaces binary pec_mask; Stage 2 keeps it alive.
        pec_mask = None

    # Build sources and probes for the compiled runner
    from rfx.sources.port_drive import PortSourceQueue
    sources = PortSourceQueue()
    mag_sources = []
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

    if sim._tfsf is not None:
        from rfx.sources.tfsf import tfsf_from_declaration

        tfsf = tfsf_from_declaration(
            sim._tfsf, grid, sim._freq_max, n_steps=n_steps,
            line_steps=decay_max_steps if until_decay is not None else n_steps,
        )
        from rfx.boundaries.tfsf import boundary_flags as tfsf_boundary_flags
        periodic, cpml_axes = tfsf_boundary_flags(tfsf[0], grid)

    # #931 §1.7: realize (Mx, My, Mz) ONCE, under the flags the stepper
    # will use, and keep working on THOSE from here on — port clearing,
    # wire-port liveness, the sheet ctx and the run all read the same
    # object.  A sheet owns no cell, so it exists only here.
    from rfx.model.conductors import kernel_conductors, clear_conductor_edges, lumped_port_stage, publish_port_release, at_kernel
    _pec_periodic = _simulation.resolve_periodic(grid, periodic)
    pec_sheets = tuple(pec_sheets or ())
    pec_wires = tuple(pec_wires or ())
    conductors = kernel_conductors(sim, grid, base_materials, pec_mask,
        pec_sheets, pec_wires, periodic=_pec_periodic, sheet_specs=sheet_specs or (), root=conductors)
    pec_edge_masks = conductors.pec_edges
    _msl_geometry_edges = pec_edge_masks  # before ANY port clearing
    if sim._msl_ports and use_kottke_pec:
        from rfx.boundaries.pec import PEC_INV_THRESHOLD, kottke_fenced_edge_masks
        # Observe the operator the stepper will actually apply: partial
        # volume edges are released, while declared sheets/wires and every
        # tensor-frozen component remain metal. This observation must not
        # change the masks passed to the stepper or turn Kottke into stairs.
        frozen = tuple(v < PEC_INV_THRESHOLD for v in aniso_inv_eps)
        if _msl_geometry_edges is None:
            _msl_geometry_edges = frozen
        else:
            fenced = kottke_fenced_edge_masks(
                _msl_geometry_edges, aniso_inv_eps, sheets=pec_sheets,
                wires=pec_wires, periodic=_pec_periodic)
            _msl_geometry_edges = tuple(m | f for m, f in zip(fenced, frozen))

    # Port sources — fold impedances into materials first
    lumped_ports = []
    wire_ports = []
    # Issue #764: per-wire-port static LIVE-cell runs and excite flags,
    # parallel to ``wire_ports`` — captured inside the loop where the
    # assembled-geometry edge masks (the #318 "live" definition, read BEFORE
    # this port's own clearing) is in hand.  Feed the S-param specs below.
    wire_port_live_cells = []
    wire_port_excites = []
    for _port_index, pe in enumerate(sim._ports):
        if pe.impedance == 0.0:
            # The declared kind alone decides the drive (forward() calls the same).
            from rfx.sources.port_drive import defer_soft_source
            defer_soft_source(sources, grid, pe, sizing_n, field_dtype=field_dtype)
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
                radius=pe.radius,
            )
            wire_ports.append(wp)
            wire_port_excites.append(bool(pe.excite))
            # Issue #764: capture the static LIVE run for this port's
            # whole-port gap-voltage accumulator while the mask still holds
            # the assembled-geometry state (pre-clearing).
            from rfx.sources.sources import _wire_port_live_cells
            _wp_cells_764, _wp_live_764, _ = _wire_port_live_cells(
                grid, wp, pec_edge_masks)
            wire_port_live_cells.append(tuple(
                (int(c[0]), int(c[1]), int(c[2]))
                for c, live in zip(_wp_cells_764, _wp_live_764) if live))
            conductors = publish_port_release(conductors, wire_port_live_cells[-1],
                component=pe.component, entity_id=f"port[{_port_index}]", kind="wire")
            # Live-cell-aware fold + injection (issue #318): dead extent
            # cells inside PEC carry no port sigma and no source. The mask
            # here is the assembled-geometry state BEFORE this port's own
            # clearing below, which is exactly the "live" definition.
            materials = setup_wire_port(grid, wp, materials,
                                        pec_edge_masks=pec_edge_masks)
            if pe.excite:
                sources.defer(_simulation.make_wire_port_sources,
                    grid, wp, n_steps=sizing_n,
                    pec_edge_masks=pec_edge_masks)
            # No PEC clearing for a wire port (#931 §1.9, corrected):
            # a cell is LIVE exactly when the port component's own edge at
            # that index is not PEC, so releasing that component at the
            # live cells is a no-op by construction, and releasing the two
            # TANGENTIAL edges there would open the conductor the port
            # foot stands on (a ground plane, a body's face).  Dead extent
            # cells stay shorted and carry no port sigma and no source.
        else:
            # Single-cell lumped port
            lp = LumpedPort(
                position=pe.position, component=pe.component,
                impedance=pe.impedance, excitation=pe.waveform,
            )
            lumped_ports.append(lp)
            materials = setup_lumped_port(grid, lp, materials)
            if pe.excite:
                sources.defer(_simulation.make_port_source, grid, lp, n_steps=sizing_n)
            # Release the realized PEC edge the lumped port DRIVES — its
            # own component at its own cell, and nothing else (#931 §1.9,
            # corrected: the three-component form opened the conductor the
            # port sits on).
            conductors = lumped_port_stage(conductors, pe, f"port[{_port_index}]")
            pec_edge_masks = conductors.pec_edges

    # One wire port and no lumped port: S11 is read from the main run's own
    # record (the fast path below), so s_param_n_steps cannot set its length.
    if (len(wire_ports) == 1 and not lumped_ports
            and compute_s_params is not False):
        sim._refuse_unsupported_run_kwargs(
            "uniform single-wire-port S-parameter", {
                "s_param_n_steps": sim._s_param_n_steps_off_record(
                    s_param_n_steps, n_steps, until_decay),
            },
            instead=None,
            reason_overrides={"s_param_n_steps":
                "this lane reads S11 from the main run's port record, "
                "whose length is n_steps, or the decay stop under until_decay"})

    from rfx.measurement.port_registration import uniform_run_specs
    record_specs = uniform_run_specs(grid, wire_ports, wire_port_live_cells, lumped_ports, sim._ports)

    from rfx.measurement.port_s_contract import passive_record_diagnostic
    passive_diagnostic = passive_record_diagnostic(sim)
    if (passive_diagnostic and compute_s_params is None and s_param_freqs is None
            and len(wire_ports) == 1 and not lumped_ports):
        # Direct runner calls do not pass through run()'s request resolver.
        from rfx.runners._admission import refuse_plain_sources_s_matrix
        compute_s_params = refuse_plain_sources_s_matrix(
            sim, main_record=True, implicit=True)
    # Main-record lumped spectra use the same bins as run's S driver.
    lumped_sparam_specs = []
    if lumped_ports and (compute_s_params is None or compute_s_params or passive_diagnostic):
        from rfx.measurement import register_lumped, default_frequency_grid
        sp_freqs = jnp.asarray(s_param_freqs if s_param_freqs is not None else
                              default_frequency_grid(sim._freq_max, 50), dtype=jnp.float32)
        for component, _cells, kind, mid, port_id in record_specs:
            if kind == 'lumped':
                pe = sim._ports[port_id]
                register_lumped([], lumped_sparam_specs, component, mid,
                                sp_freqs, pe.impedance, pe.excite, port_id)

    # Build wire port S-param specs for JIT-integrated DFT
    wire_sparam_specs = []
    if wire_ports and (compute_s_params is None or compute_s_params or passive_diagnostic):
        from rfx.simulation import WirePortSParamSpec
        sp_freqs = s_param_freqs if s_param_freqs is not None else jnp.linspace(
            sim._freq_max / 10, sim._freq_max, 50)
        # Issue #764: the V/I reference cell is the midpoint of the LIVE
        # run (a dead extent cell reads a quenched Ampere loop —
        # |I_dead|/|I_mid| = 0.003-0.03 measured on the #313 thru), and the
        # spec carries the static live run for the whole-port gap-voltage
        # accumulator.  With no dead cells this midpoint is bit-identical
        # to the historical all-extent midpoint.
        for wp, live_cells, wp_excite in zip(
                wire_ports, wire_port_live_cells, wire_port_excites):
            mid = live_cells[len(live_cells) // 2]
            wire_sparam_specs.append(WirePortSParamSpec(
                mid_i=mid[0], mid_j=mid[1], mid_k=mid[2],
                component=wp.component,
                freqs=jnp.asarray(sp_freqs, dtype=jnp.float32),
                impedance=wp.impedance,
                live_cells=live_cells,
                excite=wp_excite,
            ))

    # MSL ports — full cross-section distributed feed
    if getattr(sim, "_msl_ports", None):
        from rfx.sources.msl_port import (
            _msl_yz_cells,
            msl_normal_component as _msl_normal_component,
            compute_msl_mode_profile,
            make_msl_port_sources,
            deferred_msl_sources_jm,
            msl_cell,
            msl_cross_section_span,
            msl_port_from_entry,
            setup_msl_port,
        )
        for _msl_port_index, pe in enumerate(sim._msl_ports):
            # Issue #661: one shared projection of position -> port frame.
            mp = msl_port_from_entry(pe)
            from rfx.sources.msl_port import validate_msl_port_geometry
            validate_msl_port_geometry(
                grid, mp, pec_edge_masks=_msl_geometry_edges,
                conductors=conductors, periodic=_pec_periodic,
                pec_faces=sim._boundary_spec.pec_faces(), name=pe.name)
            mode_profile = None
            eigenmode_data = None
            port_mode = getattr(pe, "mode", "uniform")
            if port_mode in ("eigenmode", "laplace"):
                # Substrate eps_r: use explicit value if provided,
                # otherwise read from the FDTD eps_r array directly
                # under the trace centre (most representative cell).
                span = msl_cross_section_span(grid, mp)
                k_mid = (span["n_lo"] + span["n_hi"]) // 2
                eps_cell = msl_cell(
                    pe.direction, span["i_feed"], span["w_centre"], k_mid
                )
                if pe.eps_r_sub is not None:
                    eps_r_sub = float(pe.eps_r_sub)
                else:
                    eps_r_sub = float(np.asarray(materials.eps_r[eps_cell]))
            if port_mode == "eigenmode":
                from rfx.sources.msl_eigenmode import compute_msl_eigenmode_profile
                # Build frequency array from grid for β(ω) curve
                freqs_for_em = np.linspace(
                    sim._freq_max / 10.0, sim._freq_max, 20)
                eigenmode_data = compute_msl_eigenmode_profile(
                    grid, mp, eps_r_sub, freqs_for_em)
                # Also build the legacy mode_profile dict for setup_msl_port
                # so σ loading matches the eigenmode footprint.
                mode_profile = compute_msl_mode_profile(grid, mp, eps_r_sub)
            elif port_mode == "laplace":
                # Explicit static-Laplace Ez-only mode (no J+M).
                mode_profile = compute_msl_mode_profile(grid, mp, eps_r_sub)
            materials = setup_msl_port(grid, mp, materials,
                                       mode_profile=mode_profile)
            if pe.excite and pe.waveform is not None:
                if eigenmode_data is not None:
                    # Schelkunoff J+M one-sided launch: both E and H sources
                    sources.defer(deferred_msl_sources_jm, grid, mp,
                                  n_steps=sizing_n, eigenmode_data=eigenmode_data,
                                  magnetic_sources=mag_sources)
                else:
                    sources.defer(make_msl_port_sources,
                        grid, mp, n_steps=sizing_n,
                        mode_profile=mode_profile,
                    )
            # Only the SUBSTRATE-NORMAL component over the
            # cross-section: that is the edge the modal source drives
            # (V = sum(E_n * d_n)).  Releasing the two in-plane
            # components as well would punch a width-long slot through
            # the ground plane and the trace at the feed column
            # (#931 §1.9, corrected).
            conductors = clear_conductor_edges(
                conductors, list(_msl_yz_cells(grid, mp)),
                component=_msl_normal_component(mp),
                entity_id=f"msl_port[{_msl_port_index}]", port_kind="microstrip",
                released_edges=(mode_profile["cell_indices"] if mode_profile is not None else list(_msl_yz_cells(grid, mp))))
            pec_edge_masks = conductors.pec_edges

    for pe in sim._probes:
        probes.append(_simulation.make_probe(grid, pe.position, pe.component))

    axis_to_index = {"x": 0, "y": 1, "z": 2}
    for pe in sim._dft_planes:
        axis_idx = axis_to_index[pe.axis]
        plane_pos = [0.0, 0.0, 0.0]
        plane_pos[axis_idx] = pe.coordinate
        grid_index = grid.position_to_index(tuple(plane_pos))[axis_idx]
        from rfx.measurement import default_frequency_grid
        freqs_arr = (
            pe.freqs
            if pe.freqs is not None
            else default_frequency_grid(sim._freq_max, pe.n_freqs)
        )
        dft_planes.append(
            init_dft_plane_probe(
                axis=axis_idx,
                index=grid_index,
                component=pe.component,
                freqs=freqs_arr,
                grid_shape=grid.shape,
                dft_total_steps=sizing_n,
                region=getattr(sim, "_dft_plane_regions", {}).get(pe.name),
            )
        )

    if sim._waveguide_ports:
        from rfx.preflight.waveguide import resolve_waveguide_port_freqs

        cpml_axes = grid.cpml_axes
        pec_axes = None
        for pe in sim._waveguide_ports:
            freqs_arr = resolve_waveguide_port_freqs(sim, pe)
            # The port applies its incident table by step index and holds its
            # last entry past its end: sized by sizing_n (above).
            waveguide_ports.append(sim._build_waveguide_port_config(pe, grid, freqs_arr, sizing_n))

    if sim._tfsf is not None:
        if not sim._tfsf.closed_box:
            sim._validate_tfsf_vacuum_boundary(materials, tfsf[0])
        # #404: an oblique (2D-aux) TFSF drives the shared solver on the complex
        # Bloch-envelope path; final `state` and point-probe time series are
        # reconstructed to physical fields after the run, but the streaming/
        # frequency-domain output paths are not yet transform-aware on it.
        from rfx.sources.tfsf import is_tfsf_2d as _is_tfsf_2d_ru
        if _is_tfsf_2d_ru(tfsf[0]):
            _ob_unsup = []
            if snapshot is not None:
                _ob_unsup.append("snapshot= (read the final `state` instead)")
            if until_decay is not None:
                _ob_unsup.append("until_decay= (use a fixed n_steps)")
            if _ob_unsup:
                raise NotImplementedError(
                    "Oblique (angle_deg != 0) TFSF uses the complex Bloch "
                    "path; not yet supported on it: " + "; ".join(_ob_unsup)
                    + ". Supported: final `state` and `add_probe` time series "
                    "(returned as physical fields), or `compute_rcs` for "
                    "open-domain oblique scattering."
                )

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

        # A soft source at the centre of the port plane (one builder, read
        # again by the source-end collector).
        from rfx.floquet import floquet_port_source
        sources.append(floquet_port_source(grid, fpe, sim._freq_max, sim._domain, sizing_n)[1])

    from rfx.model.materials import with_components
    from rfx.geometry.rasterize_grid import unpadded_high_walls
    materials = with_components(materials, grid,
        high_wall=unpadded_high_walls(grid),
        periodic=_simulation.resolve_periodic(grid, periodic),
        debye_spec=debye_spec, lorentz_spec=lorentz_spec)
    from rfx.model.source_coefficients import resolve_run_sources
    sources = resolve_run_sources(sources, materials, sim, grid,
        tensor=aniso_eps is not None or conformal_weights is not None)

    aniso_eps = finish_tensor_operands(materials, aniso_eps, conformal_weights)

    # Thread the field storage dtype into the ADE carry (issue #656), the
    # same way ``rfx.simulation.run`` threads it into the CPML psi / NTFF /
    # RLC carries.  ``None`` here means "default precision", which
    # ``rfx.simulation.run`` resolves to float32 — resolve it identically so
    # ``precision="float32"`` keeps a float32 ADE carry even when x64 has
    # been enabled elsewhere in the process, and ``precision="mixed"``
    # (float16) gets the float32 accumulation floor it promises.
    _, debye, lorentz = sim._init_dispersion(
        materials, grid.dt, debye_spec, lorentz_spec,
        grid=grid,
        field_dtype=field_dtype if field_dtype is not None else jnp.float32,
        # The run's flags after the TFSF override above (#1260: the edge
        # mean wraps on a periodic axis, as the E update's does).
        periodic=_simulation.resolve_periodic(grid, periodic))

    # NTFF box
    ntff_box = None
    if sim._ntff is not None:
        corner_lo, corner_hi, freqs = sim._ntff
        ntff_box = make_ntff_box(grid, corner_lo, corner_hi, freqs)

    # In-loop block current moments (rfx.current_moments). Built here rather
    # than at declaration time so the slab window and the block map come from
    # the grid the solve actually builds, not from the declared corners.
    flux_monitors = build_flux_monitor_cfgs(sim, grid, sizing_n, periodic=_simulation.resolve_periodic(grid, periodic))
    current_moments = _cm_for_sim(sim, grid, periodic, conductors=conductors)

    # Lumped RLC elements
    rlc_metas = None
    if sim._lumped_rlc:
        # The run's own periodic flags (after the TFSF override above): a
        # series element's D0 is its edge's E-update coefficient (#1163).
        _rlc_periodic = _simulation.resolve_periodic(grid, periodic)
        rlc_metas = [build_rlc_meta(grid, spec, materials, periodic=_rlc_periodic)
                     for spec in sim._lumped_rlc]
        # #1245: at most one element with its own solve per realized edge.
        from rfx.lumped import refuse_stacked_solved_elements
        refuse_stacked_solved_elements(sim._lumped_rlc, rlc_metas)

    # #677: assemble the surface-impedance sheet ctx against the FINAL
    # realized PEC edges of this run (after the port clearing above —
    # PEC wins on overlapping edges). Crossing-normal refusal lives in the
    # builder.
    sheet_ctx = conductors.sheet_context(pec_edge_masks)
    if sim._tfsf is not None and sim._tfsf.closed_box:
        from rfx.boundaries.tfsf import validate_realized_vacuum_boundary
        validate_realized_vacuum_boundary(
            sim, grid, materials, tfsf[0], pec_edge_masks=pec_edge_masks,
            aniso_eps=aniso_eps, aniso_inv_eps=aniso_inv_eps,
            conformal_weights=conformal_weights, debye_spec=debye_spec,
            lorentz_spec=lorentz_spec, kerr_chi3=kerr_chi3, sheet_ctx=sheet_ctx,
            rlc_indices=((m.i, m.j, m.k) for m in (rlc_metas or ())))

    # Every declared input this lane does not carry is refused here, after
    # the specific refusals above and before the first step.
    from rfx.runners._admission import admit
    admit(
        sim,
        "run_uniform",
        run_args={"compute_s_params": compute_s_params, "conformal_pec": conformal_pec},
        diagnostics=diagnostics,
    )
    from rfx.runners._admission import admit_run_s_matrix
    admit_run_s_matrix(sim, compute_s_params=compute_s_params,
                       conformal_pec=conformal_pec)
    # Issue #1254: the early stop reads the record so far through the scan's
    # chunk hook, with the bins the single-wire fast path below reports.
    _stop_kwargs = {}
    if stop_fn is not None:
        if until_decay is not None:
            raise NotImplementedError(
                "run_uniform(stop_fn=...) does not combine with until_decay: "
                "they are two rules for where the record ends.")
        from types import SimpleNamespace as _NS

        def _stop_with_bins(steps_done, peek):
            def partial():
                p = peek()
                wps = p.wire_port_sparams
                bins = (np.array(s_param_freqs) if s_param_freqs is not None
                        else np.array(wps[0][0].freqs) if wps else None)
                return _NS(time_series=p.time_series, wire_port_sparams=wps,
                           grid=p.grid, dt=p.dt, freqs=bins)
            return stop_fn(steps_done, partial)

        _stop_kwargs = {"stop_fn": _stop_with_bins,
                        "stop_interval": int(stop_interval)}

    # The record and the call below read the same final conductor object.
    conductors, geometry_record = at_kernel(sim, conductors, lane="run_uniform", pec_edges=pec_edge_masks, sheet_operator=sheet_ctx)
    pec_edge_masks = conductors.pec_edges
    sheet_ctx = conductors.sheet_operator

    # Main simulation
    if until_decay is not None:
        # The energy stop reads W only after every drive built above has ended.
        from rfx.measurement.decay import advisory_inputs
        from rfx.measurement.record_end import drive_end
        n_end = (drive_end(grid, decay_max_steps, sources=sources, mag_sources=mag_sources,
                           tfsf=tfsf, waveguide_ports=waveguide_ports)
                 if until_decay > 0 else None)
        sim_result = _simulation.run_until_decay(
            grid, materials,
            decay_by=until_decay,
            n_end=n_end,
            advisory=advisory_inputs(sim, dt=grid.dt, n_table=decay_max_steps),
            check_interval=decay_check_interval,
            min_steps=decay_min_steps,
            max_steps=decay_max_steps,
            decay_energy_consecutive=decay_energy_consecutive,
            radiated_flux_box=radiated_flux_box,
            flux_env_checks=flux_env_checks,
            monitor_component=decay_monitor_component,
            monitor_position=decay_monitor_position,
            boundary=sim._boundary,
            cpml_axes=cpml_axes,
            pec_axes=pec_axes,
            periodic=periodic,
            debye=debye,
            lorentz=lorentz,
            tfsf=tfsf, _feature_tfsf=True,
            sources=sources,
            probes=probes,
            dft_planes=dft_planes,
            record_dft=bool(getattr(sim, "_internal_probe_indices", ())),
            flux_monitors=flux_monitors,
            waveguide_ports=waveguide_ports,
            ntff=ntff_box,
            current_moments=current_moments,
            snapshot=snapshot,
            checkpoint=checkpoint,
            aniso_eps=aniso_eps,
            aniso_inv_eps=aniso_inv_eps,
            pec_mask=pec_mask,
            pec_edge_masks=pec_edge_masks,
            pec_sheets=pec_sheets,
            pec_wires=pec_wires,
            conformal_weights=conformal_weights,
            wire_port_sparams=wire_sparam_specs or None,
            lumped_port_sparams=lumped_sparam_specs or None,
            port_record_specs=record_specs,
            lumped_rlc=rlc_metas,
            kerr_chi3=kerr_chi3,
            field_dtype=field_dtype,
            mag_sources=mag_sources or None,
            stencil_order=sim._stencil_order,
            sheet_impedance=sheet_ctx,
            **({} if report_every is None else
               {"report_every": report_every, "report_label": report_label}),
        )
    else:
        sim_result = _simulation.run(
            grid, materials, n_steps,
            boundary=sim._boundary,
            cpml_axes=cpml_axes,
            pec_axes=pec_axes,
            periodic=periodic,
            debye=debye,
            lorentz=lorentz,
            tfsf=tfsf, _feature_tfsf=True,
            sources=sources,
            probes=probes,
            dft_planes=dft_planes,
            record_dft=bool(getattr(sim, "_internal_probe_indices", ())),
            flux_monitors=flux_monitors,
            waveguide_ports=waveguide_ports,
            ntff=ntff_box,
            current_moments=current_moments,
            snapshot=snapshot,
            checkpoint=checkpoint,
            aniso_eps=aniso_eps,
            aniso_inv_eps=aniso_inv_eps,
            pec_mask=pec_mask,
            pec_edge_masks=pec_edge_masks,
            pec_sheets=pec_sheets,
            pec_wires=pec_wires,
            conformal_weights=conformal_weights,
            wire_port_sparams=wire_sparam_specs or None,
            lumped_port_sparams=lumped_sparam_specs or None,
            port_record_specs=record_specs,
            lumped_rlc=rlc_metas,
            kerr_chi3=kerr_chi3,
            field_dtype=field_dtype,
            mag_sources=mag_sources or None,
            stencil_order=sim._stencil_order,
            sheet_impedance=sheet_ctx,
            **({} if report_every is None else
               {"report_every": report_every, "report_label": report_label}),
            **_stop_kwargs,
        )

    # S-parameters: use JIT-integrated DFT for wire ports (fast),
    # fall back to Python loop for lumped ports
    if compute_s_params is None:
        compute_s_params = len(lumped_ports) > 0 or len(wire_ports) > 0

    # (2,4) stencil fence: the lumped/wire S-parameter extractors below run a
    # SEPARATE eager FDTD pass (extract_s_matrix_wire /
    # compute_lumped_wire_s_matrix_via_scan) that does not thread
    # stencil_order — under order=4 they would silently produce a 2nd-order
    # S-matrix while the main field evolution is 4th order. Reject loudly
    # rather than return a mixed-order result.
    if compute_s_params and sim._stencil_order == 4 and (lumped_ports or wire_ports):
        raise NotImplementedError(
            "run(compute_s_params=True) is not supported with "
            "stencil_order=4: the lumped/wire S-parameter extractor runs a "
            "separate 2nd-order FDTD pass that does not honour the (2,4) "
            "stencil. Use stencil_order=2 for S-parameter runs, or compute "
            "the main field at order=4 with compute_s_params=False."
        )

    s_params = None
    freqs_out = None
    s_settling = None

    from rfx.measurement.port_s import main_record_port_s
    from rfx.sparams._record_selection import SReadPorts
    main_s = main_record_port_s(
        sim, sim_result, compute_s_params=compute_s_params,
        s_param_freqs=s_param_freqs,
        off_record=sim._s_param_n_steps_off_record(s_param_n_steps, n_steps, until_decay))
    sparam_read_ids = SReadPorts()
    if main_s is not None:
        s_params, freqs_out, sparam_read_ids = main_s
    elif compute_s_params and (lumped_ports or wire_ports):
        if s_param_freqs is None:
            s_param_freqs = jnp.linspace(
                sim._freq_max / 10, sim._freq_max, 50,
            )
        freqs_out = np.array(s_param_freqs)

        # Default to the realized main record, including an until_decay stop.
        sp_n_steps = drive_steps(sim_result, n_steps, s_param_n_steps,
                                 until_decay=until_decay, decay_max_steps=decay_max_steps)

        # item-5 Stage 4 (2026-07-03): lumped AND multi-port wire run() paths go
        # through the production-scan driver (= forward()'s graph). Stage 3
        # (#214) rerouted only lumped; this completes it for MULTI-port wire, so
        # the eager extract_s_matrix / extract_s_matrix_wire loops are OFF the
        # run() hot path (the #203/#205/#206 config-drift-class root — a
        # hand-maintained second FDTD loop). They are KEPT for diagnostics /
        # openEMS-crossval tooling. For multi-port wire this also FIXES the
        # latent off-diagonal drop of the single-port fast-path above (its
        # diagonal-only fill returned S21 = 0; the driver's
        # decompose_wire_s_matrix fills the full N-port matrix, byte-close to
        # the eager extractor on CPML — tests/unit/sparams/test_sparam_driver_matches_eager).
        # (The well-conditioned single-port wire case stays on the fast-path so
        # the run()↔forward() PEC contract is preserved.)
        from rfx.probes.sparam_driver import (
            compute_lumped_wire_s_matrix_via_scan,
        )
        s_params, _, s_settling = compute_lumped_wire_s_matrix_via_scan(
            sim, s_param_freqs, n_steps=sp_n_steps,
            conformal_pec=conformal_pec,
            return_settling=True,
            skip_preflight=True,
            subpixel_smoothing=subpixel_smoothing,
        )

    declared_measurements = package_declared_measurements(
        sim, grid,
        flux_monitors=sim_result.flux_monitors or (),
        waveguide_cfgs=sim_result.waveguide_ports or (),
    )

    # #404: oblique 2D-aux TFSF evolved the complex Bloch envelope P; convert the
    # returned state + point probes back to physical real fields (real path and
    # normal incidence are untouched — sim_result stays real there).
    if sim._tfsf is not None:
        from rfx.sources.tfsf import is_tfsf_2d as _is_tfsf_2d_out
        if _is_tfsf_2d_out(tfsf[0]):
            sim_result = reconstruct_oblique_physical(sim_result, tfsf[0], grid, probes)

    # until_decay: one record of what ended the run (rfx.measurement.decay.finish_decay).
    _stop = getattr(sim_result, "decay_stop", None)
    return Result(
        diagnostics=tuple(diagnostics) + ((_stop,) if _stop is not None else ()),
        realized_geometry=geometry_record,
        state=sim_result.state,
        time_series=sim_result.time_series,
        s_params=s_params,
        freqs=freqs_out,
        ntff_data=sim_result.ntff_data,
        ntff_box=ntff_box,
        current_moment_data=sim_result.current_moment_data,
        current_moment_monitor=current_moments,
        dft_planes=(
            {entry.name: probe for entry, probe in zip(sim._dft_planes, sim_result.dft_planes or ())}
            if sim._dft_planes
            else None
        ),
        **declared_measurements,
        snapshots=sim_result.snapshots,
        snapshot_axes=sim_result.snapshot_axes,
        grid=grid,
        # The step the scan advanced by: stencil_order=4 derates it below
        # grid.dt, and every time / frequency read from this Result
        # (find_resonances, an FFT of time_series) needs the real one.
        dt=sim_result.dt,
        sparam_time_records=sim_result.sparam_time_records,
        sparam_read_ids=sparam_read_ids,
        dft_time_records={
            entry.name: rec for entry, rec in zip(sim._dft_planes, sim_result.dft_time_records or ())
        },
        settling_db=s_settling["db"] if s_settling else None,
        settling_witness=s_settling,
        freq_range=(sim._freq_max / 10, sim._freq_max, sim._boundary),
        wire_port_sparams=(sim_result.wire_port_sparams if keep_wire_port_sparams else None),
        lumped_port_sparams=sim_result.lumped_port_sparams,
    )
