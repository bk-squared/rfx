"""Multi-GPU distributed FDTD runner using jax.jit + shard_map.

Replaces the pmap-based distributed.py with the modern JAX sharding API:
- jax.sharding.Mesh + NamedSharding for explicit device placement
- jax.experimental.shard_map for ghost cell exchange and device-conditional ops
- jax.jit with in_shardings / out_shardings for the top-level compiled scan

Uses 1D slab decomposition along the x-axis with ghost cell exchange via
lax.ppermute inside shard_map.  Supports PEC and CPML boundaries, soft
sources, closed-box TF/SF plane waves, point probes, lumped ports, and
dispersive materials (Debye / Lorentz / Drude).

A direct one-device call uses the same ``shard_map`` path on a one-device mesh.
Unsupported TF/SF kinds and waveguide ports are refused before stepping.

Declared PEC volumes, sheets and sub-cell wires share the conductor object's
edge masks. Each slab applies only owned rows after source injection and
before E ghost exchange; the high-side slab owns a seam node plane.
``Simulation.run(devices=...)`` routes here only for ``len(devices) > 1``.

Source limitations (named refusals, without a one-device reroute):
- Open-slab TF/SF needs periodic y/z, which this lane does not implement.
- Method B, the 2-D Bloch line, face-record replay and modal ports do not run.
- Non-divisible nx: automatically padded to nearest multiple of n_devices
  with PEC-filled cells, then trimmed after gathering.

The public entry point ``run_distributed`` has an identical signature to the
retired pmap version so callers retain the same entry-point arguments.
"""

from __future__ import annotations

from rfx._declarations import release_declaration_caches
from rfx.errors import RfxUnsupported


from rfx.runners._distributed_common import (
    init_cpml_slabs,
)

from rfx.runners._exchange_interval import validate_exchange_interval

from rfx._internal import _realized
from rfx.stepping.slab import Slab

import jax
import jax.numpy as jnp

import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from rfx.core.yee import (
    FDTDState,
    MaterialArrays,
)
from rfx.simulation import (
    make_source,
    make_j_source,
    make_probe,
    make_port_source,
)
from rfx.sources.sources import LumpedPort, setup_lumped_port
from rfx.sources.port_drive import PortSourceQueue

# Domain splitting, local Yee update and x-slab CPML helpers. They lived in
# ``distributed.py`` until its pmap runner was retired; the names this module
# does not call are still bound here so callers that import them from it keep
# working.
from rfx.runners._distributed_common import (  # noqa: F401 -- compatibility re-exports
    gather_array_x,
    _split_state,
    _split_materials,
    _split_debye_coeffs,
    _split_debye_state,
    _split_lorentz_coeffs,
    _split_lorentz_state,
    _update_h_local,
    magnetic_low_ghost,
    _update_e_local_with_dispersion,
    _distributed_boundary_layers,
)
from rfx.runners._distributed_common import (  # noqa: F401 -- compatibility re-exports
    apply_pec_face_shmap,
    apply_pec_mask_shmap,
    _slab_curl_kwargs,
    exchange_component_shmap,
    exchange_h_yee_shmap,
    exchange_e_yee_shmap,
    inject_sources_shmap,
    sample_probes_shmap,
    shard_stacked,
    slab_e_component_materials,
    slab_e_materials_shmap,
    stage_dispersion_slabs,
    split_array_x,
    unstack_and_gather,
)


# ---------------------------------------------------------------------------
# Sharding helpers
# ---------------------------------------------------------------------------

def _make_mesh(devices):
    """Create a 1-D x-axis mesh from a list of JAX devices."""
    return Mesh(np.array(devices), axis_names=("x",))


def _spans_other_processes(devices):
    """True when a device in ``devices`` belongs to another JAX process.

    Arrays sharded over such a mesh are not fully addressable here, so the
    scan bodies must take them as jit arguments and the result needs a
    cross-process gather. On one process this is always False.
    """
    me = jax.process_index()
    return any(getattr(d, "process_index", me) != me for d in devices)


def _x_sharding(mesh):
    """NamedSharding that distributes the leading (x) axis across devices."""
    return NamedSharding(mesh, P("x"))


def _rep_sharding(mesh):
    """NamedSharding for replicated (non-distributed) arrays."""
    return NamedSharding(mesh, P())


def _shard_field_state(state: FDTDState, mesh: Mesh) -> FDTDState:
    """Place all FDTDState arrays onto the mesh, sharding along x."""
    shd = _x_sharding(mesh)
    return FDTDState(
        ex=jax.device_put(state.ex, shd),
        ey=jax.device_put(state.ey, shd),
        ez=jax.device_put(state.ez, shd),
        hx=jax.device_put(state.hx, shd),
        hy=jax.device_put(state.hy, shd),
        hz=jax.device_put(state.hz, shd),
        step=jax.device_put(state.step, _rep_sharding(mesh)),
    )


def _shard_materials(materials: MaterialArrays, mesh: Mesh) -> MaterialArrays:
    from rfx.sources.wire_radius import require_radius_update
    require_radius_update(materials, lane="distributed_v2 material shard", unsupported=True)
    if materials.sigma_film is not None:
        raise ValueError("distributed_v2 material shard does not place a lossy film; "
                         "run_distributed cuts sigma_film through rfx.stepping.slab.cut")
    shd = _x_sharding(mesh)
    return MaterialArrays(
        eps_r=jax.device_put(materials.eps_r, shd),
        sigma=jax.device_put(materials.sigma, shd),
        mu_r=jax.device_put(materials.mu_r, shd),
    )


# ---------------------------------------------------------------------------
# Ghost cell exchange via shard_map + ppermute
# ---------------------------------------------------------------------------

# Keep the shared full-exchange alias for existing callers and the NU lane.
_exchange_component_shmap = exchange_component_shmap

# Only the tangential, one-sided Yee neighbours are live on this lane.
_exchange_h_ghosts_shmap = exchange_h_yee_shmap
_exchange_e_ghosts_shmap = exchange_e_yee_shmap


# ---------------------------------------------------------------------------
# Device-conditional PEC inside shard_map
# ---------------------------------------------------------------------------

# ``_apply_pec_shmap`` aliases
# ``apply_pec_face_shmap`` in ``_distributed_common.py``.
_apply_pec_shmap = apply_pec_face_shmap



# ---------------------------------------------------------------------------
# CPML inside shard_map
# ---------------------------------------------------------------------------





# ---------------------------------------------------------------------------
# Sharded array initialisation helpers for CPML state
# ---------------------------------------------------------------------------

def _init_cpml_sharded(grid, nx_local, n_devices, mesh):
    """Place the shared kernel's per-rank state directly on the x mesh."""
    return init_cpml_slabs(grid, nx_local, n_devices, mesh=mesh)


# ---------------------------------------------------------------------------
# Public runner
# ---------------------------------------------------------------------------

def refuse_distributed_periodic(sim, *, lane, bloch=None):
    """Periodic faces cannot be replaced by a wall or an interior absorber."""
    if lane == "distributed non-uniform forward()":
        reason = "periodic faces would be treated as walls or absorbers."
        remedy = "Remove the periodic axes / Bloch phase; non-uniform periodic execution is not implemented."
    else:
        reason = ("the lane would use the declared non-periodic wall instead "
                  "(rfx.runners.distributed._update_h_local / _update_e_local).")
        single_device_hint = (
            "omit devices=... (use a single-device run() instead)"
            if lane == "distributed multi-device run()"
            else "omit distributed=True (use a single-device forward() instead)"
            if lane == "distributed multi-device forward()"
            else "call sim.run(...) without devices= instead of calling this runner"
        )
        remedy = f"Remove the periodic axes / Bloch phase, or {single_device_hint}."
    periodic_axes = getattr(sim, "_periodic_axes", "") or ""
    if bloch is None:
        bloch = getattr(sim, "_bloch", None)
    if periodic_axes or bloch is not None:
        features = []
        if periodic_axes:
            features.append("periodic axes " + ", ".join(repr(a) for a in periodic_axes))
        if bloch is not None:
            features.append("a Bloch phase")
        raise RfxUnsupported(
            f"{' and '.join(features)}: periodic / Bloch boundaries are not "
            f"supported on the {lane} path; {reason} {remedy}"
        )


def refuse_unsupported_distributed_features(sim, *, lane, bloch=None, diagnostics=(), allow_msl_run=False):
    """Refuse what the multi-device lanes would silently drop or get wrong.

    Periodic/Bloch boundaries, reference-plane or passive ports, Kerr materials,
    MSL ports, subgridding, and surface monitors (flux,
    DFT planes; NTFF on graded meshes). ``tests/contracts/path_disposition.py``
    records the per-path disposition of every Simulation attribute;
    ``tests/unit/runners/test_path_disposition_cells.py`` enforces it.

    Call before sharding.
    ``bloch`` also accepts an explicit phase from a direct caller.
    """
    single_device_hint = (
        "omit devices=... (use a single-device run() instead)"
        if lane == "distributed multi-device run()"
        else "omit distributed=True (use a single-device forward() instead)"
        if lane == "distributed multi-device forward()"
        else "call sim.run(...) without devices= instead of calling this runner"
    )
    from rfx.runners._admission import refuse_distributed_sources
    refuse_distributed_sources(sim, lane)
    forward = lane == "distributed multi-device forward()"
    refuse_distributed_periodic(sim, lane=lane, bloch=bloch)

    ports = getattr(sim, "_ports", ()) or ()
    if any(pe.extent is not None and pe.reference_plane_cells is not None
           for pe in ports):
        raise RfxUnsupported(
            "Wire ports with reference_plane_cells are not supported with "
            f"{'distributed=True' if forward else 'devices=...'}; "
            "the V_ref path needs pre-injection recordings. Use one device.")
    if any(pe.extent is not None and pe.radius is not None for pe in ports):
        from rfx.sources.wire_radius import require_radius_support
        require_radius_support(sim, "run_distributed")

    if any(pe.impedance > 0.0 and pe.extent is not None for pe in ports):
        from rfx.probes.sparam_driver import refuse_distributed_lumped_s_pmc
        refuse_distributed_lumped_s_pmc(sim, entry="forward" if forward else "run")

    if any(not pe.excite for pe in ports) and not (allow_msl_run and sim._msl_ports):
        raise RfxUnsupported(
            "add_port(..., excite=False) (passive port) is not supported "
            f"on the {lane} path; the lane would drive the port with its "
            "waveform (or raise in make_port_source for waveform=None) "
            "(rfx.runners.distributed_v2.run_distributed / "
            "rfx.runners.distributed.run_distributed port setup). "
            "Remove the passive-port configuration (excite=True), or "
            f"{single_device_hint}."
        )

    kerr = sorted(name for name, material in (getattr(sim, "_materials", None) or {}).items()
                  if getattr(material, "chi3", 0.0) != 0.0)
    if kerr:
        raise RfxUnsupported(
            f"Kerr chi3 (nonlinear) material(s) {', '.join(repr(k) for k in kerr)}: "
            f"not supported on the {lane} path; the lane drops chi3 and would run "
            "the material as linear, with no warning "
            "(rfx.runners.distributed_v2.run_distributed / "
            "rfx.runners.distributed.run_distributed material assembly). "
            f"Remove chi3, or {single_device_hint}."
        )

    msl = getattr(sim, "_msl_ports", None) or ()
    if msl and (not allow_msl_run or any(p.mode == "eigenmode" for p in msl)):
        from rfx._diagnostic_transport import diagnostic_refusal
        reason = (
            "mode='eigenmode' needs a magnetic-source channel for its J+M launch; slabs have none"
            if allow_msl_run else
            "microstrip forward has no gradient parity bar" if forward else
            "this caller does not drive or terminate microstrip ports; use compute_msl_s_matrix for microstrip S"
        )
        raise diagnostic_refusal(RfxUnsupported(
            f"add_msl_port() port(s) ({len(msl)}): not supported on the {lane} path; "
            f"{reason} (rfx.runners.distributed_v2). {single_device_hint}."
        ), diagnostics)

    if getattr(sim, "_refinement", None) is not None:
        raise RfxUnsupported(
            f"add_refinement() (subgridding): not supported on the {lane} path; "
            "the lane ignores the refinement and would solve the unrefined grid, "
            "with no warning (rfx.runners.distributed_v2.run_distributed / "
            "rfx.runners.distributed.run_distributed have no subgrid coupling). "
            "Remove the refinement, or "
            + ("use run() on one device (forward() does not take add_refinement())"
               if forward else single_device_hint) + "."
        )



def run_distributed(sim, *, diagnostics=(), n_steps, devices=None, exchange_interval=1, lane='run_distributed',
                    _source_port_indices=None, _record_probes=None, preflight=None, assembly=None,
                    snapshot=None, _record_vref=True, s_param_freqs=None, compute_s_params=False, **kwargs):
    """Run FDTD simulation distributed across multiple devices.

    Uses 1D slab decomposition along the x-axis.  Supports PEC and
    CPML boundaries, soft sources, point probes, lumped ports, and
    Debye/Lorentz dispersive materials.

    Closed-box TFSF runs on slabs. Other TFSF sources and waveguide ports
    are refused before stepping. Excited wire ports without radius or
    reference planes run.

    Parameters
    ----------
    sim : Simulation
        The Simulation instance.
    n_steps : int
        Number of timesteps.
    devices : list of jax.Device or None
        If None, use all available devices.
    exchange_interval : int, optional
        Ghost exchange interval in timesteps; only integer 1 is supported.

    _source_port_indices : tuple[int, ...] or None
        Optional excited lumped-port indices; all terminations remain present.
        Selects the S driver's source assembly (including impedance-0 sources).
        The numerical step is shared with the main run.
    _record_probes : tuple or None
        Optional global-cell probes in place of the model probes.
    s_param_freqs : array or None
        Frequency bins for in-loop main-record port accumulators.
    _record_vref : bool
        Test control: suppress only the published reference time channel.

    Returns
    -------
    Result
    """
    from rfx._diagnostic_transport import merge_diagnostics, report_diagnostics
    diagnostics = merge_diagnostics(diagnostics, report_diagnostics(assembly))
    validate_exchange_interval(exchange_interval)
    from rfx.materials.thin_conductor import refuse_f0_sheets as _refuse_f0
    from rfx.model.wall_loss import refuse_wall_loss
    refuse_wall_loss(sim, "distributed (v2) runner")
    _refuse_f0(sim._thin_conductors, "distributed (v2) runner")
    if sim._boundary == "upml":
        raise ValueError("boundary='upml' does not support distributed execution")

    refuse_unsupported_distributed_features(
        sim,
        lane="distributed (v2) runner",
        allow_msl_run=lane == "run_distributed",
        bloch=kwargs.get("bloch"),
        diagnostics=diagnostics,
    )

    from rfx.runners._distributed_records import refuse_record_options
    refuse_record_options(sim, kwargs)

    from rfx.api import Result

    if devices is None:
        devices = jax.devices()
    n_devices = len(devices)
    from rfx.api._dispatch import _staged_by_an_outer_trace
    from rfx.core.jax_utils import _REGION
    if (n_devices > 1 and sim._tfsf is not None and sim._tfsf.closed_box
            and (_REGION.get() is not None or _staged_by_an_outer_trace())):
        raise NotImplementedError(
            "Multi-device closed-box TF/SF cannot run inside a user jax.jit; "
            "use the plain call to forward() or run().")
    from rfx.runners._admission import refuse_slab_tfsf_devices
    refuse_slab_tfsf_devices(sim._tfsf, n_devices)
    # The multi-process path is needed only when the mesh holds a device
    # this process cannot address: the final fields must be gathered across
    # hosts, and non-uniform grids are not supported across processes.
    # ``jax.process_count() > 1`` is the wrong predicate -- a multi-process
    # job whose mesh is its own local devices is fully addressable, and
    # ``process_allgather`` would concatenate such arrays across processes
    # instead of replicating them.
    multi_process = _spans_other_processes(devices)

    # Resolve PMC faces. ``BoundarySpec.pmc_faces()`` returns
    # a set; freeze it so it is safely closed-over by the traced scan
    # bodies below.
    _pmc_faces_frozen = frozenset(
        sim._boundary_spec.pmc_faces()
        if getattr(sim, "_boundary_spec", None) is not None
        else ()
    )
    _pec_faces_frozen = frozenset(
        sim._boundary_spec.pec_faces()
        if getattr(sim, "_boundary_spec", None) is not None
        else ()
    )

    from rfx.core.yee import CurlBoundary

    # One device runs the same shard_map path on a one-device mesh.
    # Simulation.run(devices=...) dispatches here only for len(devices) > 1; a
    # one-device run() takes the uniform lane.

    # ------------------------------------------------------------------
    # Build grid and materials (full domain)
    # ------------------------------------------------------------------
    if sim._uses_nonuniform_mesh:
        raise NotImplementedError(
            "distributed_v2 is a uniform-grid runner; use sim.run(devices=...) "
            "for a non-uniform grid (shared distributed NU staging).")
    # Drop any standalone diagnostic arrays before allocating solve products.
    release_declaration_caches(sim)
    from rfx.model.conductors import assembled_materials, solve_conductors, kernel_conductors, at_kernel
    _d_pec_sheets: list = []
    _d_pec_wires: list = []
    _geometry_masks, _assembly_entries = [], []
    if assembly is None:
        grid = sim._build_grid()
        conductors = solve_conductors(sim, grid, n_devices=n_devices)
        if preflight is not None:
            diagnostics = merge_diagnostics(
                diagnostics, report_diagnostics(sim._auto_preflight(conductors=conductors, **preflight))
            )
    else:
        conductors = assembly.take(n_devices=n_devices)
        grid = conductors.grid
    base_materials, debye_spec, lorentz_spec, pec_mask, pec_shapes, *_assembly_rest = (
        assembled_materials(conductors, pec_sheets=_d_pec_sheets,
                                pec_wires=_d_pec_wires,
                                   geometry_masks=_geometry_masks, assembly_entries=_assembly_entries)
    )
    # The rest (Kerr chi3 among it, refused above) is unused on this lane;
    # drop it now so no whole-domain array stays alive through the loop.
    del _assembly_rest
    conductors = kernel_conductors(sim, grid, base_materials, pec_mask,
        _d_pec_sheets, _d_pec_wires, periodic=sim._periodic_flags(), root=conductors)
    del _geometry_masks, _assembly_entries
    from rfx.sources.wire_radius import require_radius_update
    require_radius_update(base_materials, lane="distributed_v2", unsupported=True)
    # Every declared input this lane does not carry is refused here, after
    # the specific refusals above, and before the first step.
    from rfx.runners._admission import admit
    admit(sim, "run_distributed", run_args={"conformal_pec": kwargs.get("conformal_pec")})
    from rfx.stepping import slab_lumped
    materials = slab_lumped.fold(sim, grid, base_materials)

    _distributed_boundary_layers(
        grid, n_devices,
        pec_faces=(_pec_faces_frozen if sim._boundary == "cpml" else
                   {f"{axis}_{side}" for axis in "xyz" for side in ("lo", "hi")}),
        pmc_faces=_pmc_faces_frozen,
        cpml_layers=grid.cpml_layers if sim._boundary == "cpml" else 0,
    )
    nx, ny, nz = grid.shape
    layout = Slab(nx, n_devices)
    pad_x = layout.pad_x
    nx_per, ghost, nx_local = layout.nx_per_rank, layout.ghost_width, layout.nx_local

    # ------------------------------------------------------------------
    # Create mesh
    # ------------------------------------------------------------------
    mesh = _make_mesh(devices)

    # ------------------------------------------------------------------
    # Build sources and probes
    # ------------------------------------------------------------------
    from rfx.measurement.prepared import attach_measurement
    attach_measurement(sim, conductors, n_steps=n_steps, path=lane,
                       frequencies=s_param_freqs if compute_s_params else None)
    from rfx._internal._grid_metric import field_index
    sources = PortSourceQueue()
    probes = []
    port_idx = -1
    wire_edges = None
    port_entries, port_specs = [], []
    msl_geometry_edges = conductors.pec_edges
    from rfx.model.conductors import lumped_port_stage, publish_port_release
    for _port_index, pe in enumerate(sim._ports):
        if pe.component not in ("ex", "ey", "ez"):
            raise ValueError(f"distributed_v2: unsupported source component {pe.component}")
        if pe.impedance > 0.0 and pe.extent is None:
            port_idx += 1
            lp = LumpedPort(
                position=pe.position, component=pe.component,
                impedance=pe.impedance, excitation=pe.waveform,
                allow_in_absorber=pe.allow_in_absorber,
            )
            mid = tuple(int(i) for i in field_index(grid, pe.position, pe.component))
            port_entries.append(pe)
            port_specs.append((pe.component, (mid,), 'lumped', mid, _port_index))
            materials = setup_lumped_port(grid, lp, materials)
            conductors = lumped_port_stage(conductors, pe, f"port[{_port_index}]")
            if pe.excite and (_source_port_indices is None or port_idx in _source_port_indices):
                sources.defer(make_port_source, grid, lp, n_steps=n_steps)
        elif pe.impedance > 0.0 and pe.extent is not None:
            from rfx.sources.sources import wire_port_from_entry, setup_wire_port
            from rfx.simulation import make_wire_port_sources
            port_idx += 1
            wp = wire_port_from_entry(pe)
            wire_edges = conductors.pec_edges
            from rfx.sources.sources import _wire_port_live_cells
            cells, live, _ = _wire_port_live_cells(grid, wp, wire_edges)
            live_cells = tuple(c for c, on in zip(cells, live) if on)
            port_entries.append(pe)
            port_specs.append((pe.component, live_cells, 'wire',
                               live_cells[len(live_cells) // 2], _port_index))
            conductors = publish_port_release(conductors, [c for c, on in zip(cells, live) if on],
                component=pe.component, entity_id=f"port[{_port_index}]", kind="wire")
            materials = setup_wire_port(grid, wp, materials, pec_edge_masks=wire_edges)
            if pe.excite and (_source_port_indices is None or port_idx in _source_port_indices):
                sources.defer(make_wire_port_sources,
                              grid, wp, n_steps=n_steps, pec_edge_masks=wire_edges)
        elif pe.is_soft_source:
            # Deferred like the ports: a load declared after this source is
            # on the edge before the drive reads it.
            if sim._boundary == "cpml":
                sources.defer(make_j_source, grid, pe.position, pe.component,
                              pe.waveform, n_steps,
                              amplitude_kind=pe.amplitude_kind)
            else:
                sources.defer(make_source, grid, pe.position, pe.component,
                              pe.waveform, n_steps,
                              amplitude_kind=pe.amplitude_kind)
    if sim._msl_ports:
        from rfx.model.msl_stamp import realize_msl_ports
        msl = realize_msl_ports(sim, grid, materials, conductors,
            eps_source=materials.eps_r, stamp_drive_materials=None,
            edge_release="run", eigenmode="run_distributed",
            geometry_edges=msl_geometry_edges, sources=sources, n_steps=n_steps)
        materials, conductors = msl.materials, msl.conductors
    if _record_probes is None:
        for pe in sim._probes:
            probes.append(make_probe(grid, pe.position, pe.component))
    else:
        # Global cells enter the ordinary owner mapping below independently.
        probes.extend(_record_probes)

    rlc_metas = slab_lumped.metadata(sim, grid, materials,
                                     pec_edges=conductors.pec_edges, uniform=True)
    from rfx.stepping.uniform_slab_ports import UniformPortPlan
    order = sorted(range(len(port_specs)), key=lambda k: port_specs[k][2] != 'wire')
    if compute_s_params and s_param_freqs is None:
        from rfx.measurement import default_frequency_grid
        s_param_freqs = default_frequency_grid(sim._freq_max, 50)
    port_args = (grid, [port_entries[k] for k in order], [port_specs[k] for k in order],
                 CurlBoundary(_pec_faces_frozen, _pmc_faces_frozen, sim._periodic_flags()))
    port_freqs = s_param_freqs if compute_s_params else None
    ports = UniformPortPlan(*port_args, jnp.float32, port_freqs)

    # All port loads must be present before any drive reads its E operands.
    # No whole-domain realization here: the slab kernels form their own edge
    # means, so the drive reads the four cells around its edge directly (this
    # path refuses periodic axes); M3 gives the slab kernel the realized rule.
    # A dispersive edge's drive needs the ADE update's coefficient, as on one
    # device; the pole terms are read edge by edge from the specs.
    from rfx.model.source_coefficients import dispersive_drive_model
    drive_materials = dispersive_drive_model(materials, debye_spec, lorentz_spec, grid=grid,
                                            in_program=mesh.devices.size < 2)
    sources = sources.resolve(drive_materials)
    del wire_edges, drive_materials

    if _realized.ACTIVE is not None:
        _realized.sources(grid, materials, sources, "distributed.sources")
    # ------------------------------------------------------------------
    from rfx.stepping import prepared as prepared_step
    conductors = prepared_step.stage(conductors, grid=grid, materials=materials)
    stepping_product = conductors
    conductors, geometry_record = at_kernel(sim, conductors, lane="run_distributed", pec_edges=conductors.pec_edges)
    pec_edges = conductors.pec_edges
    if sim._tfsf is not None and sim._tfsf.closed_box:
        from rfx.boundaries.tfsf import validate_realized_vacuum_boundary
        from rfx.sources.tfsf_discrete_plane_wave import box_bounds
        # Final whole-grid operands, after port stamping/PEC release, before
        # cutting slabs or preparing the incident line. Both public entries
        # share this runner. Other operator families are refused by admission.
        # Stateful RLC has no material stamp; include its realized edges too.
        validate_realized_vacuum_boundary(
            sim, grid, materials, box_bounds(grid.shape, grid.cpml_layers, sim._tfsf.margin),
            pec_edge_masks=pec_edges, debye_spec=debye_spec, lorentz_spec=lorentz_spec,
            rlc_indices=((m.i, m.j, m.k) for m in rlc_metas))
    del conductors, base_materials
    # Create state directly on its owning devices and stage materials one
    # addressable slab at a time, without whole-domain slab stacks.
    # ------------------------------------------------------------------

    from rfx.runners._distributed_records import SlabRecords
    from rfx.measurement.prepared import finish_measurement
    n_prb = len(probes)
    from rfx.stepping.slab_wrap_uniform import run_wrapped_uniform
    tfsf = None
    if sim._tfsf is not None:
        from rfx.stepping.slab_tfsf import prepare
        tfsf = prepare(sim, grid, n_steps, layout, mesh)
    prepared_step.stage(stepping_product, grid=grid, materials=materials, pec_edges=pec_edges)
    dispersion = [debye_spec, lorentz_spec]
    del materials, pec_edges, pec_mask, pec_shapes, debye_spec, lorentz_spec
    records = SlabRecords(sim, grid, nx_per, mesh, jnp.float32, n_steps, snapshot)
    finish_measurement(stepping_product)
    final_carry, probe_ts, final_state_sharded, records = prepared_step.advance(
        stepping_product, run_wrapped_uniform, sources, probes, layout, mesh, records,
        _operand_layout="wrapped_uniform_slab", n_steps=n_steps,
        boundary=sim._boundary, dispersion=dispersion, port_plan=ports,
        record_vref=_record_vref, tfsf=tfsf, exchange_interval=exchange_interval,
        rlc_metas=rlc_metas)

    if multi_process:
        from jax.experimental import multihost_utils

        # Return the full global fields and trace on every process. The
        # arrays are not fully addressable here (that is what selected this
        # path), so ``process_allgather`` replicates them rather than
        # concatenating per-process copies. It returns host numpy: the
        # multi-process result is not differentiable through this gather.
        # The single-process path has no gather and stays traceable.
        final_state_sharded, probe_ts = multihost_utils.process_allgather(
            (final_state_sharded, probe_ts), tiled=True,
        )

    # ------------------------------------------------------------------
    # Gather final state: sharded (n_devices*nx_local, ny, nz) ->
    # stacked (n_devices, nx_local, ny, nz) -> gathered (nx, ny, nz)
    # ------------------------------------------------------------------
    def _unstack_and_gather(sharded_arr):
        # Stage 1.5b: must remain traceable under jax.grad. The prior
        # ``np.array(sharded_arr)`` host-pull raised
        # TracerArrayConversionError whenever a caller wrapped the runner
        # in ``jax.grad`` to drive an objective from the gathered
        # ``final_state``. Pure JAX reshape + ``gather_array_x`` (already
        # JAX-friendly) keeps the gather in the trace. Shares the function
        # with ``distributed_nu.py::_unstack_and_gather``.
        return unstack_and_gather(
            sharded_arr, n_devices, nx_local, ghost, pad_x, nx,
        )

    final_state = FDTDState(
        ex=_unstack_and_gather(final_state_sharded.ex),
        ey=_unstack_and_gather(final_state_sharded.ey),
        ez=_unstack_and_gather(final_state_sharded.ez),
        hx=_unstack_and_gather(final_state_sharded.hx),
        hy=_unstack_and_gather(final_state_sharded.hy),
        hz=_unstack_and_gather(final_state_sharded.hz),
        step=jnp.asarray(final_state_sharded.step, dtype=jnp.int32),
    )

    # ------------------------------------------------------------------
    # Probe time series: summed across devices after the scan inside jit
    # probe_ts shape: (n_steps, n_probes) or (n_steps, 0)
    # ------------------------------------------------------------------
    if n_prb > 0:
        time_series = jnp.array(probe_ts)  # (n_steps, n_probes)
    else:
        time_series = jnp.zeros((n_steps, 0), dtype=jnp.float32)

    record_result = records.assemble(final_carry["records"], multi_process=multi_process)

    return Result(
        diagnostics=diagnostics,
        realized_geometry=geometry_record,
        **record_result,
        **ports.assemble(final_carry['ports'], multi_process=multi_process),
        state=final_state,
        time_series=time_series,
        s_params=None,
        freqs=None,
        grid=grid,
        dt=grid.dt,
        freq_range=(sim._freq_max / 10, sim._freq_max, sim._boundary),
    )
