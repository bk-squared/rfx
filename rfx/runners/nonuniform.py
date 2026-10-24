"""Non-uniform grid run path extracted from Simulation."""

from __future__ import annotations

from rfx._grid_metric import field_index
import warnings
from dataclasses import replace

import jax
import numpy as np
import jax.numpy as jnp

from rfx.grid import C0
from rfx.core.jax_utils import is_tracer
from rfx.core.yee import MaterialArrays, add_lumped_eps, permittivity_without_lumped
from rfx.materials.debye import init_debye
from rfx.materials.lorentz import init_lorentz
from rfx.model.overrides import apply_material_overrides
from rfx.model.waveguide_modes import _node_span_to_cell_span
from rfx.sources.port_drive import port_drive_waveform
from rfx.nonuniform import (
    NonUniformGrid,
    interior_cells,
    make_nonuniform_grid,
    run_nonuniform,
    run_nonuniform_until_decay,
    make_current_source,
)


INTERFACE_EPS_RULES = ("sampled", "dual_average")


def _validate_interface_eps_nu(sim, *, subpixel_smoothing=False, eps_override=None):
    """Refuse combinations that cannot consume the component eps rule."""
    if getattr(sim, "_interface_eps", "sampled") == "sampled":
        return
    from rfx.core.jax_utils import is_tracer

    if subpixel_smoothing or eps_override is not None:
        raise ValueError("interface_eps='dual_average' cannot combine with subpixel_smoothing or eps_override")
    if sim._thin_conductors or sim._lumped_rlc:
        raise ValueError("interface_eps='dual_average' cannot combine with thin conductors or lumped RLC")
    if any(is_tracer(getattr(sim, name, None))
           for name in ("_dx_profile", "_dy_profile", "_dz_profile")):
        raise ValueError("interface_eps='dual_average' requires concrete profiles, not traced profiles")


def assemble_interface_eps_nu(sim, grid, materials):
    """Cell-centre eps averaged over the four cells sharing each E edge.

    Tangential weights are dual-face areas, with PEC cells excluded. Along
    its own axis the edge stays in its own cell (no node-aligned interface
    cuts that edge). End nodes are one-sided. All-PEC edges retain sampled
    eps as a finite fallback; the existing PEC mask enforces their fields.
    Scalar materials, conductivity and source normalization are untouched.
    """
    from rfx.geometry.rasterize_grid import GridCoords, coords_from_nonuniform_grid, rasterize_geometry

    _validate_interface_eps_nu(sim)
    nodes = coords_from_nonuniform_grid(grid)
    widths = (grid.dx_arr_f64, grid.dy_arr_f64, grid.dz_f64)
    if any(d is None for d in widths):
        raise ValueError("interface_eps='dual_average' requires the exact float64 grid spine")
    centres = [np.asarray(x) + np.asarray(d) / 2 for x, d in zip(nodes[:3], widths)]
    # The bounding node has no outgoing real cell: copy the last centre.
    for x in centres:
        x[-1] = x[-2]
    from rfx.geometry.smoothing import continued_conductor_shape
    geometry = [replace(entry, shape=continued_conductor_shape(
                    sim, grid, entry.shape, entry=entry, unextendable=[]))
                if sim._resolve_material(entry.material_name).sigma >= sim._PEC_SIGMA_THRESHOLD
                else entry for entry in sim._geometry]
    cell_coords = GridCoords(*centres, grid.shape)
    cell, debye, lorentz, pec, *_ = rasterize_geometry(
        geometry, sim._resolve_material, cell_coords,
        centres=cell_coords, pec_sigma_threshold=sim._PEC_SIGMA_THRESHOLD)
    if debye is not None or lorentz is not None:
        raise ValueError("interface_eps='dual_average' cannot combine with Debye/Lorentz materials")
    eps = np.asarray(cell.eps_r, dtype=np.float64)
    volume_eps = np.asarray(permittivity_without_lumped(materials), dtype=np.float64)
    live = np.ones(grid.shape, dtype=np.float64) if pec is None else (~np.asarray(pec)).astype(np.float64)
    if any(not grid.is_constant(a) for a in range(3)):
        from rfx.model.materials import geometric_interface_components
        return tuple(jnp.asarray(e, dtype=materials.eps_r.dtype) for e in
                     geometric_interface_components(eps, live, volume_eps, widths))
    components = []
    for c in range(3):
        num, den = eps * live, live
        for a in range(3):
            if a == c:
                continue
            shape = [1, 1, 1]
            shape[a] = grid.shape[a]
            d = np.asarray(widths[a]).reshape(shape)
            num, den = num * d, den * d
            lower = np.maximum(np.arange(grid.shape[a]) - 1, 0)
            num = num + np.take(num, lower, axis=a)
            den = den + np.take(den, lower, axis=a)
        out = volume_eps.copy()
        np.divide(num, den, out=out, where=den > 0)
        components.append(jnp.asarray(out, dtype=materials.eps_r.dtype))
    return tuple(components)


def build_nonuniform_grid(
    freq_max: float,
    domain: tuple,
    dx: float | None,
    cpml_layers: int,
    dz_profile: np.ndarray,
    *,
    dx_profile: np.ndarray | None = None,
    dy_profile: np.ndarray | None = None,
    pec_faces: set[str] | None = None,
    pmc_faces: set[str] | None = None,
    cpml_axes: str = "xyz",
    face_layers: dict[str, int] | None = None,
    dt: float | None = None,
    dt_min_cell: float | None = None,
    dt_caller: str | None = None,
    nominal_profiles: dict | None = None,
    periodic_axes: str = "",
    automatic: bool | None = None,
    boundary_depths: tuple | None = None,
) -> NonUniformGrid:
    """Build a NonUniformGrid from per-axis profiles.

    ``dx_profile`` / ``dy_profile`` are optional; when omitted the
    corresponding axis is uniform with spacing ``dx`` across
    ``domain``. ``pec_faces`` / ``pmc_faces`` force per-face pad=0 on
    the listed faces (PMC+CPML composition fix on the NU path, 2026-04).
    ``dt`` / ``dt_min_cell`` pin a concrete time step across a mesh
    deformation family — see :func:`rfx.nonuniform.make_nonuniform_grid`.
    ``nominal_profiles`` is handed to it unchanged.
    """
    automatic = dx is None if automatic is None else automatic
    if dx is None:
        dx = C0 / freq_max / 20.0
    from rfx.grid import _periodic_resolution, cells_spanning, axis_kinds, realized_cells
    from rfx.boundaries.depths import resolve_face_depths
    depths = boundary_depths if boundary_depths is not None else resolve_face_depths(
        budget=cpml_layers, absorbing_axes=cpml_axes, pec_faces=pec_faces or (),
        pmc_faces=pmc_faces or (), periodic_axes=periodic_axes, face_layers=face_layers)
    kinds = axis_kinds(depths)
    profiles = dict(zip("xyz", (dx_profile, dy_profile, dz_profile)))
    # Unsupported profiles retain their old grid until the lane's refusal.
    periodic_axes = "".join(a for a in periodic_axes if profiles[a] is None or (
        not is_tracer(profiles[a]) and np.all(np.asarray(profiles[a]) == np.asarray(profiles[a])[0])))
    implicit = "".join(a for a in periodic_axes if profiles[a] is None)
    dx = _periodic_resolution(
        domain, implicit, dx, automatic=automatic,
        wall_axes="".join(a for a in "xyz" if kinds[a] == "wall" and profiles[a] is None))
    for axis in periodic_axes:
        profile = profiles[axis]
        if profile is not None:
            _periodic_resolution(domain, axis, float(profile[0]), automatic=False)
            length, spacing = domain["xyz".index(axis)], float(profile[0])
            if len(profile) != cells_spanning(length, spacing):
                raise ValueError(
                    f"periodic axis {axis!r}: declared period {length:.12g} m differs "
                    f"from realized period {len(profile) * spacing:.12g} m")
    if dz_profile is None:
        # Synthesise the uniform dz profile LOCALLY (pure) when only
        # dx/dy are non-uniform. This used to be done at every
        # forward()/run()/runner call site via
        # ``sim._dz_profile = np.full(...)``, permanently mutating sim
        # state as a side effect of execution (roadmap W1.3).
        nz_phys = realized_cells(domain[2], dx, kinds["z"])
        dz_profile = np.full(nz_phys, float(dx))
    domain_xy = (domain[0], domain[1])
    # An outer jit must not turn a STATIC mesh into traced coordinates:
    # conductor classification needs the same concrete lattice as preflight.
    # Actual mesh design variables retain their differentiable build path.
    from contextlib import nullcontext
    traced = any(is_tracer(value) for value in (
        dx, dx_profile, dy_profile, dz_profile))
    with nullcontext() if traced else jax.ensure_compile_time_eval():
        return make_nonuniform_grid(
            domain_xy, dz_profile, dx, cpml_layers,
            dx_profile=dx_profile, dy_profile=dy_profile,
            pec_faces=pec_faces, pmc_faces=pmc_faces, cpml_axes=cpml_axes,
            face_layers=face_layers, boundary_depths=depths,
            dt=dt, dt_min_cell=dt_min_cell, dt_caller=dt_caller,
            nominal_profiles=nominal_profiles, periodic_axes=periodic_axes, domain=tuple(map(float, domain)),
        )


def assemble_materials_nu(
    sim,
    grid: NonUniformGrid,
    sheet_specs: list | None = None,
    pec_sheets: list | None = None,
    pec_wires: list | None = None,
    geometry_masks: list | None = None,
    assembly_entries: list | None = None,
    *, n_devices: int = 1,
) -> tuple[MaterialArrays, object, object, jnp.ndarray | None]:
    """Assemble cells; preserve the four-item profiled-grid return contract."""
    from rfx.model.materials import assemble_cells
    return assemble_cells(
        sim, grid, sheet_specs=sheet_specs, pec_sheets=pec_sheets,
        pec_wires=pec_wires, geometry_masks=geometry_masks,
        assembly_entries=assembly_entries, n_devices=n_devices,
    )[:4]


def pos_to_nu_index(grid: NonUniformGrid, pos) -> tuple[int, int, int]:
    """Convert physical (x, y, z) to non-uniform grid indices.

    Delegates to ``rfx.nonuniform.position_to_index`` so the cumulative
    lookup works on non-uniform xy as well.
    """
    from rfx.nonuniform import position_to_index
    return position_to_index(grid, pos)


def _nu_flux_tangential_bounds(d_arr, pad_lo: int, pad_hi: int,
                               center, size, *, periodic=False) -> tuple[int, int]:
    """Resolve a finite graded-axis CELL window inside the physical interior.

    Uses nearest cumulative edges, with the lower edge at a tie. Every
    requested endpoint outside the interior emits a clamp warning, including
    sub-cell overflow. Ordinary interior snapping remains a geometry result.
    """
    from rfx.probes.flux_region import resolve_flux_axis

    interior = interior_cells(np.asarray(d_arr, dtype=float), pad_lo, pad_hi, periodic=periodic)
    edges = np.insert(np.cumsum(interior), 0, 0.0)
    result = resolve_flux_axis(edges, pad_lo, center, size)
    if result["clamped_low"] or result["clamped_high"]:
        warnings.warn(
            f"flux monitor center={center!r} size={size!r}: CLAMPED to interior "
            f"[0, {float(edges[-1]):.12g}] m; requested bounds "
            f"{result['requested_bounds_m']} m, realized bounds "
            f"{result['realized_bounds_m']} m. The monitor integrates the realized extent.",
            UserWarning, stacklevel=2,
        )
    return result["cell_slice"]


def _range_to_slice_nu(grid, axis, value_range):
    """Runner node aperture and physical span, shared with the geometry record."""
    d_arr_jnp = {"x": grid.dx_arr, "y": grid.dy_arr, "z": grid.dz}[axis]
    n_axis = grid.shape["xyz".index(axis)]
    pad_lo = getattr(grid, f"pad_{axis}_lo")
    pad_hi = getattr(grid, f"pad_{axis}_hi")
    d_np = np.asarray(d_arr_jnp)
    # Cell-edge positions in physical coords (interior only, edge=0 at first
    # interior face). Length = n_interior + 1.
    interior = interior_cells(d_np, pad_lo, pad_hi, periodic=axis in grid.periodic_axes)
    edges = np.insert(np.cumsum(interior), 0, 0.0)
    if value_range is None:
        return (pad_lo, n_axis - pad_hi), float(edges[-1])
    lo, hi = value_range
    # Each end on the node pos_to_nu_index names (the #1295 tie rule),
    # as the uniform builder puts it on round(lo/dx).
    from rfx.nonuniform import _axis_position_to_index
    lo_local = _axis_position_to_index(grid, axis, lo) - pad_lo
    hi_local = _axis_position_to_index(grid, axis, hi) - pad_lo
    if hi_local <= lo_local:
        raise ValueError(
            f"range {value_range!r} does not resolve to a valid aperture on the NU grid"
        )
    lo_idx = lo_local + pad_lo
    hi_idx = hi_local + pad_lo + 1
    actual_span = float(edges[hi_local] - edges[lo_local])
    if actual_span <= 0.0:
        raise ValueError(
            f"range {value_range!r} resolves to invalid aperture span {actual_span}"
        )
    return (lo_idx, hi_idx), actual_span


def _resolve_waveguide_run_freqs_nu(sim):
    """The shared bins built by the graded runner, also read by preflight."""
    for entry in sim._waveguide_ports:
        if entry.freqs is not None:
            return jnp.asarray(entry.freqs, dtype=jnp.float32)
    return jnp.linspace(sim._freq_max * 0.5, sim._freq_max, 20, dtype=jnp.float32)


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

    if normal_axis == "x":
        u_slice, a_span = _range_to_slice_nu(grid, "y", entry.y_range)
        v_slice, b_span = _range_to_slice_nu(grid, "z", entry.z_range)
    elif normal_axis == "y":
        u_slice, a_span = _range_to_slice_nu(grid, "x", entry.x_range)
        v_slice, b_span = _range_to_slice_nu(grid, "z", entry.z_range)
    else:
        u_slice, a_span = _range_to_slice_nu(grid, "x", entry.x_range)
        v_slice, b_span = _range_to_slice_nu(grid, "y", entry.y_range)

    # NODE span -> CELL span, same conversion and same reason as the uniform
    # builder (issue #868): ``_range_to_slice_nu`` reports the aperture as
    # ``edges[hi] - edges[lo]``, i.e. the cells lo..hi-1, while
    # ``WaveguidePort.u_slice`` is the field-array slice the transverse mode
    # operator is sized from.
    u_slice, v_slice = _node_span_to_cell_span(u_slice), _node_span_to_cell_span(v_slice)

    # Snapped source-plane physical coordinate (for waveguide_plane_positions).
    # Use the cumulative cell-edge position corresponding to the snapped cell
    # index along the port-normal axis.
    if normal_axis == "x":
        d_axis_np = np.asarray(grid.dx_arr)
    elif normal_axis == "y":
        d_axis_np = np.asarray(grid.dy_arr)
    else:
        d_axis_np = np.asarray(grid.dz)
    _pad_lo_axis = getattr(grid, f"pad_{normal_axis}_lo")
    _pad_hi_axis = getattr(grid, f"pad_{normal_axis}_hi")
    _interior = interior_cells(d_axis_np, _pad_lo_axis, _pad_hi_axis, periodic=normal_axis in grid.periodic_axes)
    _edges_axis = np.insert(np.cumsum(_interior), 0, 0.0)
    _local_axis = max(0, min(x_index - _pad_lo_axis, len(_edges_axis) - 1))
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
    from rfx.model.waveguide_planes import with_segment_declaration
    if entry.n_modes > 1:
        return with_segment_declaration(init_multimode_waveguide_port(
            port, grid, freqs,
            n_modes=entry.n_modes,
            f0=entry.f0 if entry.f0 is not None else sim._freq_max / 2,
            bandwidth=entry.bandwidth,
            amplitude=entry.amplitude,
            probe_offset=entry.probe_offset,
            ref_offset=entry.ref_offset,
            dft_total_steps=n_steps,
            dt=float(grid.dt),
            waveform=entry.waveform,
            mode_profile=entry.mode_profile,
        ), entry)
    return with_segment_declaration(init_waveguide_port(
        port, grid, freqs,
        f0=entry.f0 if entry.f0 is not None else sim._freq_max / 2,
        bandwidth=entry.bandwidth,
        amplitude=entry.amplitude,
        probe_offset=entry.probe_offset,
        ref_offset=entry.ref_offset,
        dft_total_steps=n_steps,
        dt=float(grid.dt),
        waveform=entry.waveform,
        mode_profile=entry.mode_profile,
    ), entry)


def _port_drive_source(grid, cell, component, excitation, n_steps, *, materials,
                      sigma_port, unit_field):
    """NU tuple source, built from the final realized materials."""
    waveform = port_drive_waveform(
        grid, cell, component, excitation, n_steps, materials,
        sigma_port=sigma_port, unit_field=unit_field)
    return (*cell, component, waveform)


def _setup_msl_ports_nu(sim, grid, materials, materials_drive, sources,
                        n_steps, pec_edge_masks, *, geometry_edge_masks=None,
                        sheet_specs=(), drawn_eps_r=None, conductors=None):
    """Set up MSL ports on the non-uniform mesh (Ez static-Laplace feed only).

    Mirrors the uniform MSL block (``rfx/runners/uniform.py``: ``_msl_ports``)
    but with two NU-specific differences:
      * the eigenmode J+M launch is FENCED — ``run_nonuniform`` carries no
        magnetic-source channel, so the Schelkunoff H-source would have
        nowhere to go; and
      * the feed's own termination conductance is stamped into
        ``materials_drive`` as well as ``materials`` before the feed is
        queued (#1256). The queue reads final realized materials after all
        stamps; ``materials_drive`` carries any whole-grid
        eps/sigma override, traced or not (#1267).

    The substrate ``eps_r`` of the launch fixture -- the static-Laplace mode
    shape, when the port does not state ``eps_r_sub`` -- is read at the
    feed's centre cell from ``drawn_eps_r``, the permittivity as drawn,
    whenever an override is in force, and from ``materials_drive`` (which is
    then the drawn one) otherwise. The fixture is a static shape: under an
    override the uniform ``forward()`` reads it from the registered
    materials too (#483), so a finite difference through the override and
    ``jax.grad`` differentiate one function, and a traced override never
    reaches the host-side Laplace solve. Only the Cb each feed cell is
    driven through follows the override.

    The Ez point-sources are appended to ``sources`` (they ride the generic
    NU point-source scan injection). The per-probe DFT planes are registered
    separately by ``compute_msl_s_matrix`` via ``add_dft_plane_probe`` and
    flow through the existing NU ``dft_plane_probes`` accumulation. Returns
    the (possibly σ-updated) ``materials`` and the port-cleared realized PEC
    edge masks (#931 §1.9).
    """
    from rfx.sources.msl_port import (
        _msl_yz_cells,
        msl_normal_component as _msl_normal_component,
        compute_msl_mode_profile,
        make_msl_port_sources,
        msl_cell,
        msl_cross_section_span,
        msl_port_from_entry,
        setup_msl_port,
    )
    from rfx.model.conductors import kernel_conductors, clear_conductor_edges
    return_object = conductors is not None
    if conductors is None:
        from dataclasses import replace
        conductors = replace(kernel_conductors(sim, grid, materials, None,
            periodic=tuple(a in grid.periodic_axes for a in "xyz"), sheet_specs=sheet_specs), pec_edges=pec_edge_masks)
    original_edges = pec_edge_masks if geometry_edge_masks is None else geometry_edge_masks
    for _msl_port_index, pe in enumerate(sim._msl_ports):
        # Issue #661: one shared projection of position -> port frame.
        mp = msl_port_from_entry(pe)
        port_mode = getattr(pe, "mode", "laplace")
        if port_mode == "eigenmode":
            raise NotImplementedError(
                "NU MSL S-params support mode='laplace'/'uniform' (Ez feed) "
                "only: the eigenmode J+M launch needs the magnetic-source "
                "channel, which run_nonuniform does not carry. Use "
                "mode='laplace' (the add_msl_port default) on the "
                "non-uniform lane."
            )
        # laplace / uniform: build the static-Laplace Ez mode profile from the
        # substrate eps_r read concretely under the trace centre.
        from rfx.sources.msl_port import validate_msl_port_geometry
        validate_msl_port_geometry(
            grid, mp, pec_edge_masks=original_edges, conductors=conductors,
            pec_faces=sim._boundary_spec.pec_faces(), name=pe.name,
            periodic=tuple(a in grid.periodic_axes for a in "xyz"))
        span = msl_cross_section_span(grid, mp)
        k_mid = (span["n_lo"] + span["n_hi"]) // 2
        eps_cell = msl_cell(
            pe.direction, span["i_feed"], span["w_centre"], k_mid
        )
        if pe.eps_r_sub is not None:
            eps_r_sub = float(pe.eps_r_sub)
        else:
            _fixture_eps = (materials_drive.eps_r if drawn_eps_r is None
                            else drawn_eps_r)
            eps_r_sub = float(np.asarray(_fixture_eps[eps_cell]))
        mode_profile = compute_msl_mode_profile(grid, mp, eps_r_sub)

        materials = setup_msl_port(grid, mp, materials, mode_profile=mode_profile)
        materials_drive = setup_msl_port(      # #1256
            grid, mp, materials_drive, mode_profile=mode_profile)
        if pe.excite and pe.waveform is not None:
            sources.defer(make_msl_port_sources,
                grid, mp, n_steps=n_steps, mode_profile=mode_profile,
            )
        # Only the SUBSTRATE-NORMAL component: the edge the modal
        # source drives.  The three-component form opened a
        # width-long slot in the ground plane at the feed (#931
        # §1.9, corrected).
        conductors = clear_conductor_edges(
            conductors, list(_msl_yz_cells(grid, mp)),
            component=_msl_normal_component(mp),
            entity_id=f"msl_port[{_msl_port_index}]", port_kind="microstrip", released_edges=mode_profile["cell_indices"])
        pec_edge_masks = conductors.pec_edges
    return materials, conductors if return_object else pec_edge_masks


def run_nonuniform_path(sim, *, diagnostics=(), n_steps, compute_s_params=None, s_param_freqs=None,
                        preflight=None, conductors=None,
                        eps_override=None, sigma_override=None,
                        pec_mask_override=None, pec_occupancy_override=None,
                        pinned_sheet_layout=None,
                        checkpoint=False,
                        emit_time_series=True, checkpoint_every=None,
                        n_warmup=0,
                        report_every=None, report_label="",
                        subpixel_smoothing: bool = False,
                        attach_waveguide_flux: bool = False,
                        strip_sheet_impedance: bool = False,
                        until_decay: float | None = None,
                        decay_check_interval: int = 50,
                        decay_min_steps: int = 100,
                        decay_max_steps: int = 50_000,
                        decay_energy_consecutive: int = 2,
                        radiated_flux_box: tuple | None = None,
                        flux_env_checks: int = 4,
                        design_box=None,
                        lane: str = "run_nonuniform",
                        stop_fn=None,
                        stop_interval: int = 250,
                        conformal_pec=None,
                        _drive_scan: bool = False):
    """Run simulation on non-uniform grid with graded dz.

    Parameters
    ----------
    sim : Simulation
        The Simulation instance (read-only access to its fields).
    n_steps : int
        Number of timesteps.
    until_decay : float or None
        When not None, run via :func:`run_nonuniform_until_decay`
        (issue #383 — NU port of the #169 interior-energy stop): every
        step-count-sized build (source tables, waveguide port record
        buffers, DFT planes, flux monitors) is sized with
        ``decay_max_steps`` instead of ``n_steps``, and the scan is
        replaced by the chunked host loop that stops when the
        dV-weighted interior energy has decayed to this fraction of
        peak. The routing layer only sends this on absorbing
        (cpml/upml) boundaries. ``checkpoint`` (the jax.checkpoint grad
        tape flag) is accepted-and-ignored on the decay path — it is a
        forward-only host loop (mirrors the uniform run_until_decay);
        ``checkpoint_every`` / ``n_warmup`` raise NotImplementedError
        below. Flux monitors must use the rectangular DFT window (the
        NU default) — a streaming windowed DFT needs the true total
        step count in advance, which a decay-terminated run does not
        have, so non-rect windows raise ValueError instead of silently
        mis-windowing.
    decay_check_interval, decay_min_steps, decay_max_steps,
    decay_energy_consecutive :
        Threaded to :func:`run_nonuniform_until_decay` (same semantics
        as ``Simulation.run``'s ``decay_*`` kwargs).
    stop_fn, stop_interval :
        ``run(..., ringdown=..., until_identified=True)`` (issue #1254): the
        ``n_steps`` record runs as :func:`run_nonuniform_until_decay`'s
        chunked loop (chunks of ``stop_interval`` steps, energy stop off,
        the #667 progress route's settings) and ``stop_fn(steps_done,
        peek)`` decides after each chunk whether it ends there; ``peek()``
        gives the record so far (``time_series``, ``wire_port_sparams``,
        ``freqs``, ``grid``, ``dt`` as this function reports them).
        ``None`` (default) is the unchanged path.
    compute_s_params : bool or None
    s_param_freqs : array or None
    eps_override, sigma_override : jnp.ndarray or None
        When provided, replace the assembled material arrays before
        source/port setup. Used by the differentiable ``forward()``
        path to inject optimisation variables.
    pec_mask_override : jnp.ndarray or None
        Extra hard-PEC mask ORed into the geometry-derived pec_mask.
    design_box : DesignBoxSpec or None
        Issue #1183. One static box whose E update is redone from its own
        (usually traced) permittivity, so reverse-mode AD keeps box-shaped
        arrays per timestep instead of grid-shaped ones. The graded-mesh
        counterpart of the uniform lane's ``rfx.simulation.run(design_box=)``
        (#1179); ``rfx.model.resolve_design_box``, called from
        ``_build_nu_scan``, carries the fences that need the resolved step
        context, and the ones below are this lane's own.
    lane : str
        The lane admission judges this run as: ``"run_nonuniform"`` (the
        default, also a direct call) or ``"fwd_nonuniform"`` from
        ``forward()``. A declared input the lane does not carry is refused
        before the first step (``rfx.runners._admission``).

    Returns
    -------
    Result
    """
    # run()'s S for the port sets uniform run() scans: one drive per port through
    # the shared driver (compute_lumped_wire_s_matrix_via_scan), not this record.
    from rfx.runners._admission import _s_matrix_ports, graded_lumped_default_s
    from rfx.measurement.port_s_contract import passive_record_diagnostic
    passive_diagnostic = passive_record_diagnostic(sim)  # all-passive set: V/I kept, as uniform
    # One excited lumped port alone reads S from the main record, as uniform run() does.
    single_lumped = (len(sim._ports) == 1 and sim._ports[0].extent is None
                     and sim._ports[0].impedance != 0.0 and sim._ports[0].excite)
    scan = (lane == "run_nonuniform" and not _drive_scan and compute_s_params is not False
            and _s_matrix_ports(sim) and not single_lumped
            and (compute_s_params is True or s_param_freqs is not None
                 or any(p.impedance != 0.0 and p.extent is not None for p in sim._ports)
                 or graded_lumped_default_s(sim)))
    if scan:
        # The main record keeps its port records at the scan's bins; its S is not formed.
        from rfx.measurement.monitors import default_frequency_grid
        compute_s_params = True
        s_param_freqs = (s_param_freqs if s_param_freqs is not None
                         else default_frequency_grid(sim._freq_max, 50, host=True))

    if not scan and (lane == "run_nonuniform" and compute_s_params is not False
            and any(p.impedance != 0.0 and (p.extent is not None or compute_s_params is True or s_param_freqs is not None) for p in sim._ports)):
        from rfx.runners._admission import refuse_plain_sources_s_matrix
        if not refuse_plain_sources_s_matrix(sim, main_record=True,
                implicit=passive_diagnostic and compute_s_params is None and s_param_freqs is None):
            compute_s_params = False

    if not scan and (compute_s_params is not False
            and any(p.impedance > 0 and (p.extent is not None or compute_s_params is True or s_param_freqs is not None) for p in sim._ports)):
        from rfx.runners._admission import require_isolated_s_drive
        if not require_isolated_s_drive(
                sim, implicit=lane == "fwd_nonuniform" and s_param_freqs is None,
                fields_only=lane == "run_nonuniform"):
            compute_s_params = False

    # every single-device non-uniform solve (run AND forward) enters here
    from rfx.sources.tfsf import _refuse_extended_tfsf
    _refuse_extended_tfsf(sim._tfsf, "the non-uniform runner")
    sim._require_mode_the_nonuniform_lane_solves()
    # compute_waveguide_s_matrix's graded branch reaches this lane without
    # _dispatch_plan, so the lane refuses the refinement it drops (#1240).
    sim._require_no_refinement_on_the_nonuniform_lane()
    from rfx.api import Result

    from rfx.runners._smoothing_admission import refuse_nonuniform_smoothing
    refuse_nonuniform_smoothing(sim, subpixel_smoothing=subpixel_smoothing,
        eps_override=eps_override, pec_occupancy=pec_occupancy_override)
    _validate_interface_eps_nu(sim, subpixel_smoothing=subpixel_smoothing,
                               eps_override=eps_override)

    # ---- #1183 design box: the fences only this lane can see ----
    # Everything the resolved step context decides (the absorber, the
    # source/wire-port/RLC cells, dispersion, subpixel eps, a sheet inside
    # the box) is rfx.model.resolve_design_box's, from _build_nu_scan.
    # These three are declarations that never reach it: a modal port writes
    # its own E over a whole plane from a mode profile, and an MSL port
    # REWRITES ``materials`` across its cross-section (_setup_msl_ports_nu)
    # -- both from the background permittivity the box no longer carries.
    if design_box is not None:
        if sim._waveguide_ports:
            raise NotImplementedError(
                "a design box (#1183) does not combine with a waveguide "
                "port on the non-uniform lane: the port injects and "
                "extracts a modal field over its whole plane, built from "
                "the background permittivity before the time loop. Use "
                "eps_override.")
        if getattr(sim, "_msl_ports", None):
            raise NotImplementedError(
                "a design box (#1183) does not combine with an MSL port on "
                "the non-uniform lane: the port rewrites ``materials`` "
                "across its cross-section at setup, which the design "
                "permittivity would not reach. Use eps_override.")
        if until_decay is not None:
            raise NotImplementedError(
                "a design box (#1183) does not combine with until_decay on "
                "the non-uniform lane: the decay stop is a forward-only "
                "host loop, so there is no gradient tape for the box to "
                "keep small. Use a fixed n_steps run.")

    # Flux monitors: the NU scan body accumulates Poynting-flux DFTs (parity
    # with the uniform path). Full-plane AND finite-region (``size=``)
    # monitors are supported; the finite-region tangential CELL window is
    # resolved against the graded cumulative cell edges in the monitor-build
    # block below (``_nu_flux_tangential_bounds``, audit c/B1, issue #764).

    # ---- until_decay (issue #383) fences + sizing ----
    if until_decay is not None:
        if checkpoint_every is not None:
            raise NotImplementedError(
                "checkpoint_every is not supported with until_decay on the "
                "non-uniform path: the decay stop drives a chunked host "
                "loop, not a single lax.scan, so segmented remat does not "
                "apply (mirrors run_until_decay's checkpoint_segments "
                "raise on the uniform lane). Use a fixed n_steps run for "
                "checkpoint_every."
            )
        if n_warmup:
            raise NotImplementedError(
                "n_warmup is not supported with until_decay on the "
                "non-uniform path: the decay stop is a forward-only host "
                "loop, so the warmup AD-tape split does not apply."
            )
        from rfx.measurement.decay import refuse_windowed_flux
        refuse_windowed_flux(sim, "non-uniform")

    # Step-count-sized builds use ``sizing_n``: on the decay path every
    # source table / waveguide-port record buffer / DFT accumulator is
    # allocated for the worst case (decay_max_steps) so nothing truncates
    # before the stop fires; stopping early is safe (the waveguide
    # post-scan rect DFT masks by n_steps_recorded, and the streaming
    # DFT-plane / flux / wire-port accumulators simply stop accumulating).
    sizing_n = int(decay_max_steps) if until_decay is not None else n_steps

    # Public run/forward lend their preflighted solve object; internal callers
    # without one still use the same builder here.
    grid = conductors.grid if conductors is not None else sim._build_nonuniform_grid()
    _sheet_specs: list = []
    _pec_sheets: list = []
    _pec_wires: list = []
    _geometry_masks, _assembly_entries = ([], []) if lane == "run_nonuniform" else (None, None)
    from rfx.model.conductors import assembled_materials, solve_conductors, kernel_conductors, prepare_port_occupancy, at_kernel
    if conductors is None:
        conductors = solve_conductors(sim, grid, nonuniform=True)
    if preflight is not None:
        from rfx._diagnostic_transport import merge_diagnostics, report_diagnostics
        diagnostics = merge_diagnostics(
            diagnostics, report_diagnostics(sim._auto_preflight(conductors=conductors, **preflight))
        )
    materials, debye_spec, lorentz_spec, pec_mask = assembled_materials(
        conductors, sheet_specs=_sheet_specs, pec_sheets=_pec_sheets,
        pec_wires=_pec_wires, geometry_masks=_geometry_masks, assembly_entries=_assembly_entries)
    del _geometry_masks, _assembly_entries
    if getattr(sim, "_interface_eps", "sampled") == "dual_average" and (
            debye_spec is not None or lorentz_spec is not None):
        raise ValueError("interface_eps='dual_average' cannot combine with Debye/Lorentz materials")
    if strip_sheet_impedance:
        # #677 operator-isolation control, used by the sheet negative
        # controls and resonance A/B lock. A sigma override cannot remove
        # this separate surface operator.
        _sheet_specs = []

    # ``eps_override`` / ``sigma_override`` replace the assembled material
    # arrays for the scan, and every source and port DRIVE is built from
    # ``materials_drive``, which starts from the SAME arrays -- overridden,
    # and traced when the override is (#1267). A drive pushes its current
    # through the Cb = dt/(eps + sigma*dt/2) of the edge it feeds, so it has
    # to read the permittivity the E update reads there. Built from the
    # arrays as drawn, as this lane did until #1267, a device whose
    # permittivity under the port came from the override was driven
    # Cb_drawn/Cb_override times too hard: a layered substrate (eps_r
    # 3.38/3.38/10.2 under a 50 ohm wire) supplied by override read S11
    # 0.027 away from the same board declared as materials, +0.065 dB at
    # 6 GHz, and any permittivity derivative taken through the override
    # carried that ratio's derivative as a false amplitude term. The
    # uniform ``forward()`` builds its drives from the traced overridden
    # materials (``rfx/api/_execute.py``); a traced override now makes this
    # lane's source table traced in the same way (``port_drive_waveform``,
    # ``make_current_source`` and ``make_msl_port_sources`` stay in jnp
    # for a tracer). A two-run
    # reference (the waveguide S-matrix's vacuum override) likewise drives
    # any current source through its own arrays; its modal port drives never
    # read them. The one thing still read from the arrays as drawn is the
    # MSL launch fixture's substrate eps_r (``_setup_msl_ports_nu``,
    # ``drawn_eps_r``): a static mode shape, as on the uniform lane (#483).
    #
    # #1256: ``materials_drive`` is not a frozen snapshot. Each port below
    # stamps its termination conductance into BOTH copies. A port drives a
    # current into an edge whose update coefficient Cb = dt/(eps + sigma*dt/2)
    # includes that port's own conductance; a drive built without it
    # injected (1 + sigma_port*dt/(2*eps)) times the declared current --
    # 2.9 on a 50 ohm port across three 0.5 mm cells of eps_r 3.38 at the
    # Courant step, so the absolute field depended on dt and on the dual
    # cell sizes at the feed. S-parameters, ratios of two responses to one
    # drive, hid it only where every cell of the port had the same factor;
    # a layered substrate, graded cells or a lumped R/C under the port
    # made the factors unequal and moved S11 too. The uniform lane builds
    # its port sources after the stamp and never had this. On a concrete mesh the stamp is a host float and the copy stays
    # concrete; on a traced mesh (#1207) sigma_port is traced, and so is the
    # drive, exactly as the E update is.
    materials_drive = materials
    _drawn_eps_r = None     # only an override separates drawn from stepped
    if eps_override is not None or sigma_override is not None:
        _drawn_eps_r = materials.eps_r
        # A whole-grid override REPLACES the array, so any lumped stamp that
        # was folded into it is gone; its #1210 record goes with it, or the
        # E update would add back a load the override does not carry.
        materials = apply_material_overrides(
            materials, eps_override=eps_override, sigma_override=sigma_override)
        materials_drive = materials     # #1267: the drive sees the override
    elif design_box is not None:
        # #1183, the same rule at the same place: the design permittivity
        # sets the precision of the material arithmetic, exactly as an
        # ``eps_override`` array does one branch up. Without it the two
        # halves of one grid step at two precisions -- the float32 rounding
        # of ``eps_r * EPS_0`` alone moves the field by ~1e-7 relative.
        materials = apply_material_overrides(materials, design_box=design_box)

    if pec_mask_override is not None:
        pec_mask = pec_mask_override if pec_mask is None else (pec_mask | pec_mask_override)

    # #931 §1.7: realize (Mx, My, Mz) ONCE here with the grid's periodic
    # flags, matching the seam convention of the 3-D NU step function.
    # Port clearing, wire-port liveness and the sheet ctx all read
    # THIS object from here on.
    _pec_sheets = tuple(_pec_sheets)
    _pec_wires = tuple(_pec_wires)
    conductors = kernel_conductors(sim, grid, materials, pec_mask,
        _pec_sheets, _pec_wires, periodic=tuple(a in grid.periodic_axes for a in "xyz"), sheet_specs=_sheet_specs, root=conductors)
    pec_edge_masks = conductors.pec_edges
    _msl_geometry_edges = pec_edge_masks  # before ANY port clearing
    from rfx.model.conductors import (
        refuse_layout_metal_consumers, refuse_layout_port_releases,
        refuse_layout_sheet_impedance,
    )
    refuse_layout_metal_consumers(pinned_sheet_layout, sim, grid)
    refuse_layout_sheet_impedance(pinned_sheet_layout, conductors)

    if subpixel_smoothing:
        from rfx.geometry.smoothing import refuse_pmc_smoothing
        refuse_pmc_smoothing(sim, grid)

    # ── Subpixel smoothing on non-uniform mesh ─────────────────────────
    # Builds Kottke tensor-averaged ε per E-component using per-axis
    # cell-size arrays. Only the non-dispersive scan branch consumes
    # ``aniso_eps`` (matches the uniform-path semantics — see
    # rfx/simulation.py::_update_e_with_optional_dispersion).
    aniso_eps = None
    if subpixel_smoothing:
        from rfx.geometry.smoothing import compute_smoothed_eps_nonuniform
        if debye_spec is not None or lorentz_spec is not None:
            sim._refuse_unsupported_run_kwargs(
                "non-uniform mesh with Debye/Lorentz materials",
                {"subpixel_smoothing": subpixel_smoothing},
                instead="remove the Debye/Lorentz poles",
                reason_overrides={"subpixel_smoothing":
                    "the dispersive E update does not read the smoothed "
                    "per-component permittivity tensor, so interfaces "
                    "would get scalar eps"},
            )
        else:
            # #1043 stage B: the pairs carry the CPML/UPML pad continuation,
            # the NU mirror of the two uniform sites. Same shared builder, so
            # the three cannot drift the way the array-side replication did
            # before #627.
            from rfx.geometry.smoothing import (
                smoothed_shape_pairs, warn_unextendable_shapes, warn_curved_high_contrast,
            )
            shape_eps_pairs, _unextendable = smoothed_shape_pairs(sim, grid)
            if subpixel_smoothing != "full_tensor":
                warn_curved_high_contrast(shape_eps_pairs, sim=sim)
            warn_unextendable_shapes([u for u in _unextendable if not u.conductor])
            if shape_eps_pairs:
                if subpixel_smoothing == "full_tensor":
                    from rfx.model.tensor_operands import full_tensor_eps
                    aniso_eps = full_tensor_eps(sim, grid, materials, shape_eps_pairs)
                else:
                    aniso_eps = compute_smoothed_eps_nonuniform(
                        grid, shape_eps_pairs, background_eps=1.0,
                    )

    if getattr(sim, "_interface_eps", "sampled") == "dual_average":
        aniso_eps = assemble_interface_eps_nu(sim, grid, materials)

    # Fold RLC R/C into materials before other port/source setup
    # (mirrors the uniform path). #1256: into the drive's copy too -- an
    # R or C across a driven edge is part of that edge's Cb, and a drive
    # built without it is off by the element's own (eps + sigma*dt/2) ratio.
    if sim._lumped_rlc:
        from rfx.lumped import setup_rlc_materials
        for spec in sim._lumped_rlc:
            materials = setup_rlc_materials(grid, spec, materials)
            materials_drive = setup_rlc_materials(grid, spec, materials_drive)

    # Smoothing / dual averaging above builds only the volume tensor.
    # Fold first, then put each capacitor on its own edge exactly once.
    from rfx.model.tensor_operands import finish_tensor_operands
    aniso_eps = finish_tensor_operands(materials, aniso_eps, None)

    from rfx.sources.port_drive import PortSourceQueue
    sources = PortSourceQueue()
    probes = []
    from rfx.model.graded_ports import assemble_graded_ports
    from rfx.runners._admission import graded_lumped_default_s
    ports = assemble_graded_ports(
        sim, grid, materials, materials_drive, conductors,
        pec_edge_masks=pec_edge_masks, sizing_n=sizing_n, lane=lane,
        s_param_freqs=s_param_freqs,
        compute_s_params=(True if lane == "run_nonuniform" and compute_s_params is None
                          and s_param_freqs is None and graded_lumped_default_s(sim)
                          else compute_s_params))
    materials, materials_drive, conductors = (
        ports.materials, ports.materials_drive, ports.conductors)
    # Wire publication leaves pec_edges unchanged; only lumped releases
    # advance them inside assembly, as in the original port loop.
    pec_edge_masks = conductors.pec_edges
    wire_port_specs = ports.port_specs
    if scan:
        for spec in wire_port_specs:
            spec['record_s'] = False
    drives_by_port = {}
    for port_index, drive in ports.drives:
        drives_by_port.setdefault(port_index, []).append(drive)
    for port_index, pe in enumerate(sim._ports):
        if pe.impedance == 0.0:
            idx = (pos_to_nu_index(grid, pe.position) if getattr(pe, "extent", None) is not None
                   else field_index(grid, pe.position, pe.component))
            sources.defer(make_current_source,
                          grid, idx, pe.component, pe.waveform, sizing_n,
                          amplitude_kind=pe.amplitude_kind)
        for cell, component, waveform, sigma_port, unit_field in drives_by_port.get(port_index, ()):
            sources.defer(_port_drive_source,
                          grid, cell, component, waveform, sizing_n,
                          sigma_port=sigma_port, unit_field=unit_field)

    for pe in sim._probes:
        idx = (pos_to_nu_index(grid, pe.position) if getattr(pe, "extent", None) is not None
               else field_index(grid, pe.position, pe.component))
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
            from rfx.measurement import default_frequency_grid
            freqs_arr = (
                pe.freqs
                if pe.freqs is not None
                else default_frequency_grid(sim._freq_max, pe.n_freqs)
            )
            dft_plane_probes.append(
                init_dft_plane_probe(
                    axis=axis_idx,
                    index=grid_index,
                    component=pe.component,
                    freqs=freqs_arr,
                    grid_shape=(grid.nx, grid.ny, grid.nz),
                    dft_total_steps=sizing_n,
                    region=getattr(sim, "_dft_plane_regions", {}).get(pe.name),
                )
            )

    # Flux monitors. Full-plane monitors integrate the entire plane;
    # finite-region (``size=``) monitors restrict the DFT accumulation to the
    # CELL window whose cumulative face area matches the requested physical
    # size, resolved against the graded cumulative cell edges via
    # ``_nu_flux_tangential_bounds`` (audit c/B1, issue #764). Tangential cell
    # sizes are passed as per-cell arrays so a graded tangential axis is
    # integrated correctly by FluxMonitor.dA.
    flux_monitor_objs = build_flux_monitor_cfgs_nu(sim, grid, sizing_n)

    sp_freqs = None
    if any(not p.get("record_only", False) for p in wire_port_specs) and (
        s_param_freqs is not None if lane == "fwd_nonuniform"
        else compute_s_params is None or compute_s_params or passive_diagnostic
    ):
        sp_freqs = s_param_freqs
        if sp_freqs is None:
            from rfx.measurement.monitors import default_frequency_grid
            sp_freqs = default_frequency_grid(sim._freq_max, 50, host=True)

    # Waveguide ports: build per-port config via NU-aware
    # init_waveguide_port (duck-types on grid for per-axis widths).
    waveguide_port_cfgs = []
    if sim._waveguide_ports:
        wg_freqs = _resolve_waveguide_run_freqs_nu(sim)
        for pe in sim._waveguide_ports:
            waveguide_port_cfgs.append(
                _build_waveguide_port_config_nu(
                    sim, pe, grid, wg_freqs, sizing_n,
                )
            )

    # MSL ports — full cross-section distributed Ez feed (NU lane). Sources
    # ride the generic point-source `sources` list; per-probe DFT planes are
    # registered by compute_msl_s_matrix via add_dft_plane_probe.
    if getattr(sim, "_msl_ports", None):
        materials, conductors = _setup_msl_ports_nu(
            sim, grid, materials, materials_drive, sources, sizing_n,
            pec_edge_masks,
            geometry_edge_masks=_msl_geometry_edges, sheet_specs=_sheet_specs,
            drawn_eps_r=_drawn_eps_r, conductors=conductors,
        )
        pec_edge_masks = conductors.pec_edges

    from rfx.model.materials import with_components
    materials = with_components(materials, grid, periodic=tuple(a in grid.periodic_axes for a in "xyz"),
        debye_spec=debye_spec, lorentz_spec=lorentz_spec)
    sources = sources.resolve(materials)

    # Lumped RLC: build per-element metadata + zero-init ADE states
    rlc_metas: tuple = ()
    rlc_states_init: tuple = ()
    if sim._lumped_rlc:
        from rfx.lumped import (build_rlc_meta, init_rlc_state,
                                refuse_stacked_solved_elements)
        rlc_metas = tuple(
            build_rlc_meta(grid, spec, materials) for spec in sim._lumped_rlc
        )
        # #1245: at most one element with its own solve per realized edge.
        refuse_stacked_solved_elements(sim._lumped_rlc, rlc_metas)
        rlc_states_init = tuple(init_rlc_state() for _ in sim._lumped_rlc)

    # Debye/Lorentz coefficients, AFTER the last stamp into ``materials``
    # (the wire, lumped and MSL port loads above), as the uniform lane builds
    # them. With a dispersive material anywhere in the model the E update runs
    # on these coefficients alone and never reads ``materials.sigma`` (#1257):
    # built before the port stamps they left every port unterminated, and a
    # 50 ohm and a 5000 ohm port gave the same waveform.
    debye = None
    if debye_spec is not None:
        debye_poles, debye_masks = debye_spec
        debye = init_debye(debye_poles, materials, grid.dt, mask=debye_masks,
                           periodic=tuple(a in grid.periodic_axes for a in "xyz"))

    lorentz = None
    if lorentz_spec is not None:
        lorentz_poles, lorentz_masks = lorentz_spec
        lorentz = init_lorentz(lorentz_poles, materials, grid.dt, mask=lorentz_masks,
                               periodic=tuple(a in grid.periodic_axes for a in "xyz"))

    # Optional per-waveguide-port Poynting flux monitors at each port's
    # probe plane (issue #88 flux-extractor path). Built from the same
    # cfg geometry the uniform ``extract_waveguide_s_matrix_flux`` uses,
    # but with per-cell tangential cell-size arrays so a graded tangential
    # axis integrates correctly via FluxMonitor.dA.
    wg_flux_monitors = []
    if attach_waveguide_flux and waveguide_port_cfgs:
        from rfx.measurement.flux_weights import grid_flux_monitor
        from rfx.boundaries.pmc import magnetic_image_faces
        _axis_idx = {"x": 0, "y": 1, "z": 2}
        for cfg in waveguide_port_cfgs:
            ax = _axis_idx[cfg.normal_axis]
            wg_flux_monitors.append(
                grid_flux_monitor(
                    grid, axis=ax,
                    index=cfg.probe_x,
                    freqs=cfg.freqs,
                    dft_total_steps=sizing_n,
                    lo1=cfg.u_lo, hi1=cfg.u_hi,
                    lo2=cfg.v_lo, hi2=cfg.v_hi,
                    pmc_faces=magnetic_image_faces(sim._boundary_spec.pmc_faces(),
                                                   (grid.nx, grid.ny, grid.nz)), staggered_area=True,
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
        if is_tracer(grid.cells("x")):
            raise ValueError(
                "a plane wave on a graded mesh needs a fixed, uniform x axis; "
                "the x cell sizes are traced")
        from rfx.sources.tfsf import tfsf_from_declaration
        tfsf_pair = tfsf_from_declaration(
            entry, grid, sim._freq_max, n_steps=n_steps,
        )
        nonvacuum = []
        if design_box is not None:
            b = design_box.bounds
            window = tuple(slice(b[2 * d], min(b[2 * d + 1] + 1, grid.shape[d]))
                           for d in range(3))
            nonvacuum.append(jnp.zeros(grid.shape, dtype=bool).at[window].set(True))
        sim._validate_tfsf_vacuum_boundary(
            materials, tfsf_pair[0], nonvacuum_masks=tuple(nonvacuum))

    # Share the box builder with the realized-geometry record, including
    # periodic extent endpoints and the face-centre sampling margin.
    ntff_box = None
    ntff_data_init = None
    if sim._ntff is not None:
        from rfx.farfield import make_ntff_box, init_ntff_data
        ntff_box = make_ntff_box(grid, *sim._ntff)
        ntff_data_init = init_ntff_data(ntff_box)

    # In-loop block current moments (rfx.current_moments), realized against
    # the grid the solve actually builds.
    #
    # The monitor receives the same effective flags as the graded step.
    from rfx.current_moments import monitor_for_simulation as _cm_for_sim
    current_moments = _cm_for_sim(
        sim, grid, periodic=tuple(a in grid.periodic_axes for a in "xyz"), conductors=conductors,
        overrides={"eps_r": eps_override, "sigma": sigma_override,
                   "pec_mask": pec_mask_override,
                   "pec_occupancy": pec_occupancy_override,
                   "design_box": design_box})

    # #677: assemble the surface-impedance sheet ctx from the specs the
    # assembler emitted, against the FINAL realized PEC edges of this run
    # (PEC wins on overlapping edges). Crossing-normal refusal lives in the
    # builder.
    sheet_ctx = conductors.sheet_context(pec_edge_masks)
    if sheet_ctx is not None:
        # v1 fences (loud, never silent): the sheet operator replaces the
        # standard E update at its edges, which is only correct against the
        # plain isotropic update_e_nu path.
        if debye_spec is not None or lorentz_spec is not None:
            raise ValueError(
                "surface-impedance (surface_impedance_f0) sheets combined "
                "with dispersive (Debye/Lorentz) materials in one run are "
                "not supported (#677 v1): the sheet operator would "
                "silently override the ADE dispersion update at its edges. "
                "Remove the dispersive material or the f0 sheet.")
        if aniso_eps is not None:
            raise ValueError(
                "surface-impedance (surface_impedance_f0) sheets combined "
                "with subpixel_smoothing / anisotropic permittivity are "
                "not supported (#677 v1): the sheet operator assumes the "
                "isotropic E update at its edges. Disable "
                "subpixel_smoothing or drop the f0 sheet.")

    # Every declared input this lane does not carry is refused here, after
    # the specific refusals above and before the first step.
    from rfx.runners._admission import admit
    from rfx.boundaries.graded_periodic import admit_graded_periodic
    admit_graded_periodic(sim)
    admit(
        sim,
        lane,
        run_args={"conformal_pec": conformal_pec, "compute_s_params": compute_s_params},
        diagnostics=diagnostics,
    )

    conductors, pec_occupancy_override = prepare_port_occupancy(
        conductors, waveguide_port_cfgs, pec_occupancy_override, None, sim=sim, sheet_operator=sheet_ctx)
    if pinned_sheet_layout:
        # The waveguide entries are published on a copy for this check only, so
        # the solved product is the one the same drawing gets without a layout.
        from rfx.model.conductors import waveguide_port_stage
        released = conductors
        for index, config in enumerate(waveguide_port_cfgs):
            released = waveguide_port_stage(released, config, f'waveguide_port[{index}]')
        refuse_layout_port_releases(pinned_sheet_layout, released)
    conductors, geometry_record = at_kernel(sim, conductors, lane=lane, pec_edges=pec_edge_masks, sheet_operator=sheet_ctx)
    pec_edge_masks = conductors.pec_edges
    sheet_ctx = conductors.sheet_operator
    _shared_run_kwargs = dict(
        design_box=design_box,
        sheet_impedance=sheet_ctx,
        aniso_eps=aniso_eps,
        pec_mask=pec_mask,
        pec_edge_masks=pec_edge_masks,
        pec_sheets=_pec_sheets,
        pec_wires=_pec_wires,
        pec_occupancy=pec_occupancy_override,
        pinned_sheet_layout=pinned_sheet_layout,
        sources=sources,
        probes=probes,
        wire_ports=wire_port_specs if wire_port_specs else None,
        s_param_freqs=sp_freqs,
        debye=debye,
        lorentz=lorentz,
        pec_faces=getattr(sim, '_pec_faces', None),
        pmc_faces=sim._boundary_spec.pmc_faces() if getattr(sim, '_boundary_spec', None) is not None else None,
        dft_planes=dft_plane_probes if dft_plane_probes else None,
        record_dft=bool(getattr(sim, "_internal_probe_indices", ())),
        flux_monitors=(
            (flux_monitor_objs + wg_flux_monitors)
            if (flux_monitor_objs or wg_flux_monitors) else None
        ),
        rlc_metas=rlc_metas,
        rlc_states=rlc_states_init,
        ntff_box=ntff_box,
        ntff_data=ntff_data_init,
        current_moments=current_moments,
        waveguide_ports=waveguide_port_cfgs if waveguide_port_cfgs else None,
        tfsf=tfsf_pair, _feature_tfsf=True,
        emit_time_series=emit_time_series,
    )
    if stop_fn is not None:
        # #1254: the chunked loop of the #667 progress route (energy stop
        # off, min_steps past the end) with the caller's stop check.
        if until_decay is not None or checkpoint_every is not None or n_warmup:
            raise NotImplementedError(
                "run_nonuniform_path(stop_fn=...) does not combine with "
                "until_decay, checkpoint_every or n_warmup (issue #1254).")
        from types import SimpleNamespace as _NS

        from rfx.progress import validate_report_every
        _re_stop = (None if report_every is None
                    else validate_report_every(report_every, n_steps=n_steps))

        def _stop_with_result(steps_done, peek):
            def partial():
                d = peek()
                return _NS(time_series=d["time_series"],
                           wire_port_sparams=d.get("wire_port_sparams"),
                           lumped_port_sparams=d.get("lumped_port_sparams"),
                           freqs=d.get("s_param_freqs"), grid=grid, dt=grid.dt)
            return stop_fn(steps_done, partial)

        r = run_nonuniform_until_decay(
            grid, materials,
            decay_by=0.0,
            check_interval=int(stop_interval),
            min_steps=n_steps + 1,
            max_steps=n_steps,
            decay_energy_consecutive=1,
            report_every=_re_stop,
            report_label=report_label,
            stop_fn=_stop_with_result,
            **_shared_run_kwargs,
        )
    elif until_decay is not None:
        # The energy stop reads W only after every drive built above has ended.
        from rfx.measurement.decay import advisory_inputs, drive_end
        n_end = (drive_end(grid, decay_max_steps, sources=sources, tfsf=tfsf_pair,
                           waveguide_ports=waveguide_port_cfgs)
                 if until_decay > 0 and radiated_flux_box is None else None)
        # #383: chunked host loop with the interior-energy stop. The
        # fences above already rejected checkpoint_every / n_warmup /
        # non-rect flux windows; ``checkpoint`` is accepted-and-ignored
        # (forward-only host loop — mirrors uniform run_until_decay's
        # treatment of the grad-tape flag).
        r = run_nonuniform_until_decay(
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
            report_every=report_every,
            report_label=report_label,
            **_shared_run_kwargs,
        )
    elif (report_every is not None and not checkpoint
          and checkpoint_every is None and not n_warmup):
        # #667 on the NU lane: drive the SAME chunked host loop the decay
        # stop uses, with the stop disabled (decay_by=0.0 is that loop's
        # documented forced-N escape) and min_steps past the end so no
        # energy check ever runs. Chunk re-entry threads the full carry
        # (bit-identity locked by tests/unit/nonuniform/test_nu_progress_chunking.py).
        from rfx.progress import validate_report_every
        _re = validate_report_every(report_every, n_steps=n_steps)
        r = run_nonuniform_until_decay(
            grid, materials,
            decay_by=0.0,
            check_interval=_re,
            min_steps=n_steps + 1,
            max_steps=n_steps,
            decay_energy_consecutive=1,
            report_every=_re,
            report_label=report_label,
            **_shared_run_kwargs,
        )
    else:
        if report_every is not None:
            import warnings
            warnings.warn(
                f"report_every={report_every} is ignored on this non-uniform "
                "run: chunked progress (#667) composes with neither "
                "checkpoint/segmented-remat nor n_warmup on the NU lane "
                "(the chunked loop is forward-only).",
                UserWarning, stacklevel=2)
        r = run_nonuniform(
            grid, materials, n_steps,
            checkpoint=checkpoint,
            checkpoint_every=checkpoint_every,
            n_warmup=n_warmup,
            **_shared_run_kwargs,
        )

    s_params = r.get("s_params")
    freqs_out = r.get("s_param_freqs")
    s_settling = None
    if scan:
        from rfx.probes.sparam_driver import compute_lumped_wire_s_matrix_via_scan
        # The main record's bins, as a record-read run() reports them.
        s_params, _, s_settling = compute_lumped_wire_s_matrix_via_scan(
            sim, freqs_out, n_steps=n_steps, conformal_pec=conformal_pec, return_settling=True,
            skip_preflight=True, subpixel_smoothing=subpixel_smoothing)

    from rfx.measurement import package_declared_measurements
    declared_measurements = package_declared_measurements(
        sim, grid,
        flux_monitors=r.get("flux_monitors", ()),
        waveguide_cfgs=r.get("waveguide_ports", ()),
    )

    # Repack DFT planes into {name: DFTPlaneProbe} dict to match
    # the uniform path's Result schema.
    dft_planes_dict = None
    if sim._dft_planes and "dft_planes" in r:
        dft_planes_dict = {
            entry.name: probe
            for entry, probe in zip(sim._dft_planes, r["dft_planes"])
        }

    # Extract per-waveguide-port Poynting flux spectra (issue #88 flux path).
    waveguide_port_flux_result = None
    if attach_waveguide_flux and wg_flux_monitors and "flux_monitors" in r:
        from rfx.probes.probes import flux_spectrum
        n_sim_flux = len(flux_monitor_objs)
        wg_final = r["flux_monitors"][n_sim_flux:]
        # jnp-native (NU flux-AD, mirrors the uniform PR #172 fix): keep
        # flux_spectrum on the AD tape — np.array() here would detach the
        # eps_override-traced gradient (the issue #148 concretization bug).
        # The sole consumer (compute_waveguide_s_matrix_nu, normalize='flux')
        # is jnp-native; np.asarray(...) downstream still accepts these for
        # any concrete-path numpy reader, so forward values are unchanged.
        waveguide_port_flux_result = tuple(
            flux_spectrum(m) for m in wg_final
        )

    # until_decay: one record of what ended the run (rfx.measurement.decay.finish_decay).
    _stop = r.get("decay_stop")
    from rfx.model.design_admission import with_design_diagnostics
    return with_design_diagnostics(Result(
        diagnostics=tuple(diagnostics) + ((_stop,) if _stop is not None else ()),
        realized_geometry=geometry_record,
        state=r["state"],
        time_series=r["time_series"],
        s_params=s_params,
        freqs=freqs_out,
        **({} if s_settling is None else dict(settling_db=s_settling["db"], settling_witness=s_settling)),
        ntff_data=r.get("ntff_data"),
        ntff_box=ntff_box,
        current_moment_data=r.get("current_moment_data"),
        current_moment_monitor=current_moments,
        dft_planes=dft_planes_dict,
        wire_port_sparams=r.get("wire_port_sparams"),
        lumped_port_sparams=r.get("lumped_port_sparams"),
        sparam_time_records=r.get("sparam_time_records"),
        sparam_read_ids=r.get("sparam_read_ids"),
        dft_time_records={
            entry.name: record for entry, record in zip(sim._dft_planes, r.get("dft_time_records", ()))
        },
        **declared_measurements,
        waveguide_port_flux=waveguide_port_flux_result,
        grid=grid,
        dt=grid.dt,
        freq_range=(sim._freq_max / 10, sim._freq_max, sim._boundary),
    ), conductors)


def build_flux_monitor_cfgs_nu(sim, grid, sizing_n):
    """Build the graded lane's flux monitors on its realized cell windows."""
    flux_monitor_objs = []
    if getattr(sim, "_flux_monitors", None):
        from rfx.measurement.flux_weights import grid_flux_monitor
        from rfx.measurement import default_frequency_grid
        from rfx.boundaries.pmc import magnetic_image_faces
        axis_to_index = {"x": 0, "y": 1, "z": 2}
        from rfx.probes.flux_region import resolve_flux_region
        for pe in sim._flux_monitors:
            axis_idx = axis_to_index[pe.axis]
            plane_pos = [0.0, 0.0, 0.0]
            plane_pos[axis_idx] = pe.coordinate
            grid_index = pos_to_nu_index(grid, tuple(plane_pos))[axis_idx]
            freqs_arr = (
                pe.freqs
                if pe.freqs is not None
                else default_frequency_grid(sim._freq_max, pe.n_freqs)
            )
            # Finite-region window (B1): physical size/center -> CELL slice on
            # each tangential axis against the graded cumulative edges. Full
            # plane keeps init_flux_monitor's (0, -1) full-extent defaults.
            lo1, hi1, lo2, hi2 = 0, -1, 0, -1
            region = resolve_flux_region(grid, pe, sim._domain)
            if region is not None:
                (lo1, hi1), (lo2, hi2) = region["cell_slices"]
                grid_index = region["normal_index"]
            flux_monitor_objs.append(
                grid_flux_monitor(
                    grid, axis=axis_idx,
                    index=grid_index,
                    freqs=freqs_arr,
                    dft_total_steps=sizing_n,
                    dft_window=getattr(pe, "dft_window", "rect"),
                    dft_window_alpha=getattr(pe, "dft_window_alpha", 0.25),
                    lo1=lo1, hi1=hi1, lo2=lo2, hi2=hi2,
                    pmc_faces=magnetic_image_faces(sim._boundary_spec.pmc_faces(),
                                                   (grid.nx, grid.ny, grid.nz)), staggered_area=True,
                )
            )

    return flux_monitor_objs
