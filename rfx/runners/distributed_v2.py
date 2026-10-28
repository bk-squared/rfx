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

from functools import partial
from types import SimpleNamespace

from rfx.boundaries.cpml import apply_cpml_e, apply_cpml_h, cpml_grid_metadata
from rfx.runners._distributed_common import (
    init_cpml_slabs, cpml_slab_frame, cpml_absorbing_faces, cpml_h_materials,
)

from rfx.runners._exchange_interval import validate_exchange_interval

from rfx import _realized
from rfx.stepping.slab import Slab, cut

import jax
import jax.numpy as jnp

from rfx.core.drives import drive_layout
from jax import lax
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from rfx.runners._rank import mesh_ranks, rank_shard_map

from rfx.core.yee import (
    FDTDState,
    MaterialArrays,
    map_lumped,
)
from rfx.simulation import (
    make_source,
    make_j_source,
    make_probe,
    make_port_source,
)
from rfx.sources.sources import LumpedPort, setup_lumped_port
from rfx.sources.port_drive import PortSourceQueue
from rfx.materials.debye import DebyeCoeffs, DebyeState
from rfx.materials.lorentz import LorentzCoeffs, LorentzState

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

# #1038 leg 3. ``_apply_pec_shmap``'s body now lives as
# ``apply_pec_face_shmap`` in ``_distributed_common.py`` (the NU runner
# carried a copy that differed only in its parameter spelling and three
# comments). Kept as a module-local alias so the existing call sites are
# unchanged.
_apply_pec_shmap = apply_pec_face_shmap



# ---------------------------------------------------------------------------
# CPML inside shard_map
# ---------------------------------------------------------------------------

from rfx.stepping.slab_cpml import _apply_cpml_e_shmap


from rfx.stepping.slab_cpml import _apply_cpml_h_shmap


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
        raise NotImplementedError(
            f"{' and '.join(features)}: periodic / Bloch boundaries are not "
            f"supported on the {lane} path; {reason} {remedy}"
        )


def refuse_unsupported_distributed_features(sim, *, lane, bloch=None, diagnostics=()):
    """Refuse what the multi-device lanes would silently drop or get wrong.

    Periodic/Bloch boundaries, reference-plane or passive ports, Kerr materials,
    lumped RLC elements, MSL ports, subgridding, and surface monitors (flux,
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
        raise NotImplementedError(
            "Wire ports with reference_plane_cells are not supported with "
            f"{'distributed=True' if forward else 'devices=...'}; "
            "the V_ref path needs pre-injection recordings. Use one device.")
    if any(pe.extent is not None and pe.radius is not None for pe in ports):
        from rfx.sources.wire_radius import require_radius_support
        require_radius_support(sim, "run_distributed")

    if any(pe.impedance > 0.0 and pe.extent is not None for pe in ports):
        from rfx.probes.sparam_driver import refuse_distributed_lumped_s_pmc
        refuse_distributed_lumped_s_pmc(sim, entry="forward" if forward else "run")

    if any(not pe.excite for pe in ports):
        raise NotImplementedError(
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
        raise NotImplementedError(
            f"Kerr chi3 (nonlinear) material(s) {', '.join(repr(k) for k in kerr)}: "
            f"not supported on the {lane} path; the lane drops chi3 and would run "
            "the material as linear, with no warning "
            "(rfx.runners.distributed_v2.run_distributed / "
            "rfx.runners.distributed.run_distributed material assembly). "
            f"Remove chi3, or {single_device_hint}."
        )

    rlc = getattr(sim, "_lumped_rlc", None) or ()
    if rlc:
        raise NotImplementedError(
            f"add_lumped_rlc() element(s) ({len(rlc)}): not supported on the {lane} "
            "path; the lane never applies them and would solve the structure as if "
            "the elements were absent, with no warning (#1239; "
            "rfx.runners.distributed_v2.run_distributed / "
            "rfx.runners.distributed.run_distributed have no lumped-element update). "
            f"Remove the elements, or {single_device_hint}."
        )

    msl = getattr(sim, "_msl_ports", None) or ()
    if msl:
        from rfx._diagnostic_transport import diagnostic_refusal
        raise diagnostic_refusal(NotImplementedError(
            f"add_msl_port() port(s) ({len(msl)}): not supported on the {lane} path; "
            "the lane never drives or terminates microstrip ports, so a model fed "
            "only by them returns a zero field, with no warning "
            "(rfx.runners.distributed_v2.run_distributed / "
            "rfx.runners.distributed.run_distributed have no MSL port update). "
            f"Remove the MSL ports, or {single_device_hint}."
        ), diagnostics)

    if getattr(sim, "_refinement", None) is not None:
        raise NotImplementedError(
            f"add_refinement() (subgridding): not supported on the {lane} path; "
            "the lane ignores the refinement and would solve the unrefined grid, "
            "with no warning (rfx.runners.distributed_v2.run_distributed / "
            "rfx.runners.distributed.run_distributed have no subgrid coupling). "
            "Remove the refinement, or "
            + ("use run() on one device (forward() does not take add_refinement())"
               if forward else single_device_hint) + "."
        )



def run_distributed(sim, *, diagnostics=(), n_steps, devices=None, exchange_interval=1,
                    _source_port_indices=None, _record_probes=None, preflight=None, assembly=None,
                    snapshot=None, _pre_injection_probes=(), _record_vref=True, **kwargs):
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
    _pre_injection_probes : tuple
        Private S-driver E probes; append their pre-injection samples to the
        probe output. _record_vref=False is the bit-identity test control.

    Returns
    -------
    Result
    """
    from rfx._diagnostic_transport import merge_diagnostics, report_diagnostics, result_with_diagnostics
    diagnostics = merge_diagnostics(diagnostics, report_diagnostics(assembly))
    validate_exchange_interval(exchange_interval)
    from rfx.materials.thin_conductor import refuse_f0_sheets as _refuse_f0
    _refuse_f0(sim._thin_conductors, "distributed (v2) runner")
    if sim._boundary == "upml":
        raise ValueError("boundary='upml' does not support distributed execution")

    refuse_unsupported_distributed_features(
        sim,
        lane="distributed (v2) runner",
        bloch=kwargs.get("bloch"),
        diagnostics=diagnostics,
    )

    from rfx.runners._distributed_records import refuse_record_options
    refuse_record_options(sim, kwargs)

    from rfx.api import Result

    if devices is None:
        devices = jax.devices()
    n_devices = len(devices)
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

    # Resolve PMC faces (T8, 2026-04). ``BoundarySpec.pmc_faces()`` returns
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
    curl_boundary = (CurlBoundary(_pec_faces_frozen, _pmc_faces_frozen, sim._periodic_flags())
                     if _pmc_faces_frozen else None)

    # One device runs the same shard_map path on a one-device mesh (#1296).
    # Until #1296 this call was handed to the pmap runner, which dropped the
    # declared-PEC cell mask and so refused a PEC volume at one device that
    # this path realizes at two. Simulation.run(devices=...) dispatches here
    # only for len(devices) > 1; a one-device run() takes the uniform lane.

    # ------------------------------------------------------------------
    # Build grid and materials (full domain)
    # ------------------------------------------------------------------
    if sim._uses_nonuniform_mesh:
        raise NotImplementedError(
            "distributed_v2 is a uniform-grid runner; use sim.run(devices=...) "
            "for a non-uniform grid (shared distributed NU staging).")
    # Drop any standalone diagnostic arrays before allocating solve products.
    sim._pf_campaign_ctx = None
    sim._realized_geometry_record = None
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
    from rfx.geometry.rasterize_grid import unpadded_high_walls
    high_wall = unpadded_high_walls(grid)
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
    materials = base_materials

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

    use_cpml = sim._boundary == "cpml" and grid.cpml_layers > 0
    n_cpml = grid.cpml_layers if use_cpml else 0

    dt = grid.dt
    dx = grid.dx

    # ------------------------------------------------------------------
    # Create mesh
    # ------------------------------------------------------------------
    mesh = _make_mesh(devices)

    # ------------------------------------------------------------------
    # Build sources and probes
    # ------------------------------------------------------------------
    sources = PortSourceQueue()
    probes = []
    port_idx = -1
    wire_edges = None
    from rfx.model.conductors import lumped_port_stage, publish_port_release
    for _port_index, pe in enumerate(sim._ports):
        if pe.component not in ("ex", "ey", "ez"):
            raise ValueError(f"distributed_v2: unsupported source component {pe.component}")
        if pe.impedance > 0.0 and pe.extent is None:
            port_idx += 1
            lp = LumpedPort(
                position=pe.position, component=pe.component,
                impedance=pe.impedance, excitation=pe.waveform,
            )
            materials = setup_lumped_port(grid, lp, materials)
            conductors = lumped_port_stage(conductors, pe, f"port[{_port_index}]")
            if _source_port_indices is None or port_idx in _source_port_indices:
                sources.defer(make_port_source, grid, lp, n_steps=n_steps)
        elif pe.impedance > 0.0 and pe.extent is not None:
            from rfx.sources.sources import wire_port_from_entry, setup_wire_port
            from rfx.simulation import make_wire_port_sources
            port_idx += 1
            wp = wire_port_from_entry(pe)
            wire_edges = conductors.pec_edges
            from rfx.sources.sources import _wire_port_live_cells
            cells, live, _ = _wire_port_live_cells(grid, wp, wire_edges)
            conductors = publish_port_release(conductors, [c for c, on in zip(cells, live) if on],
                component=pe.component, entity_id=f"port[{_port_index}]", kind="wire")
            materials = setup_wire_port(grid, wp, materials, pec_edge_masks=wire_edges)
            if _source_port_indices is None or port_idx in _source_port_indices:
                sources.defer(make_wire_port_sources,
                              grid, wp, n_steps=n_steps, pec_edge_masks=wire_edges)
        elif pe.impedance == 0.0:
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
    if _record_probes is None:
        for pe in sim._probes:
            probes.append(make_probe(grid, pe.position, pe.component))
    else:
        # Global cells enter the ordinary owner mapping below independently.
        probes.extend(_record_probes)

    # All port loads must be present before any drive reads its E operands.
    # No whole-domain realization here: the slab kernels form their own edge
    # means, so the drive reads the four cells around its edge directly (this
    # path refuses periodic axes); M3 gives the slab kernel the realized rule.
    # A dispersive edge's drive needs the ADE update's coefficient, as on one
    # device (#1524); the pole terms are read edge by edge from the specs.
    from rfx.model.source_coefficients import dispersive_drive_model
    drive_materials = dispersive_drive_model(materials, debye_spec, lorentz_spec, grid=grid,
                                            in_program=mesh.devices.size < 2)
    sources = sources.resolve(drive_materials)
    del wire_edges, drive_materials

    # Map source/probe global indices to (device_id, local_index)
    src_device_ids = []
    src_local_specs = []
    for s in sources:
        dev_id = layout.owner(s.i)
        local_i = layout.local_index(s.i)
        src_device_ids.append(dev_id)
        src_local_specs.append((local_i, s.j, s.k, s.component))

    prb_device_ids = []
    prb_local_specs = []
    for p in probes:
        dev_id = layout.owner(p.i)
        local_i = layout.local_index(p.i)
        prb_device_ids.append(dev_id)
        prb_local_specs.append((local_i, p.j, p.k, p.component))

    if _realized.ACTIVE is not None:
        _realized.sources(grid, materials, sources, "distributed.sources")
    # Precompute source waveforms: (n_steps, n_sources)
    if sources:
        src_waveforms = jnp.stack([s.waveform for s in sources], axis=-1)
    else:
        src_waveforms = jnp.zeros((n_steps, 0), dtype=jnp.float32)

    # ------------------------------------------------------------------
    conductors, geometry_record = at_kernel(sim, conductors, lane="run_distributed", pec_edges=conductors.pec_edges)
    pec_edges = conductors.pec_edges
    if sim._tfsf is not None and sim._tfsf.closed_box:
        from rfx.boundaries.tfsf import validate_realized_vacuum_boundary
        from rfx.sources.tfsf_discrete_plane_wave import box_bounds
        # Final whole-grid operands, after port stamping/PEC release, before
        # cutting slabs or preparing the incident line. Both public entries
        # share this runner. Other operator families are refused by admission.
        validate_realized_vacuum_boundary(
            sim, grid, materials, box_bounds(grid.shape, grid.cpml_layers, sim._tfsf.margin),
            pec_edge_masks=pec_edges, debye_spec=debye_spec, lorentz_spec=lorentz_spec)
    del conductors, base_materials
    # Create state directly on its owning devices and stage materials one
    # addressable slab at a time, without whole-domain slab stacks.
    # ------------------------------------------------------------------

    shd = _x_sharding(mesh)
    rep = _rep_sharding(mesh)

    field_shape = (n_devices * nx_local, ny, nz)
    sharded_state = FDTDState(
        **{name: jnp.zeros(field_shape, dtype=jnp.float32, device=shd)
           for name in ("ex", "ey", "ez", "hx", "hy", "hz")},
        step=jax.device_put(jnp.int32(0), rep),
    )

    sharded_materials = MaterialArrays(
        eps_r=cut(materials.eps_r, layout, "eps_r", mesh=mesh),
        sigma=cut(materials.sigma, layout, "sigma", mesh=mesh),
        mu_r=cut(materials.mu_r, layout, "mu_r", mesh=mesh),
        # #1236: a lumped element loads its own E edge only; the slab update
        # needs the per-component record for that (None when no stamp).
        sigma_film=cut(materials.sigma_film, layout, "lumped", mesh=mesh),
        sigma_lumped=cut(materials.sigma_lumped, layout, "lumped", mesh=mesh),
        eps_r_lumped=cut(materials.eps_r_lumped, layout, "lumped", mesh=mesh),
    )

    # #1053 leg 1. ``None`` whenever the model declares no PEC volume. The
    # scan receives it as a jit argument next to ``sharded_materials`` (on
    # every topology since PI decision A, 2026-09-23), not through
    # ``run_distributed``'s ``**kwargs``: that bag is read only for ``bloch``
    # and otherwise discarded (it was forwarded only to the pmap runner at one
    # device, until #1296).
    # Alignment rows are still PEC; ghosts are still excluded by the kernel.
    sharded_pec_mask = cut(pec_edges, layout, "pec_edge", mesh=mesh)
    del pec_edges

    # ------------------------------------------------------------------
    # Dispersive materials
    # ------------------------------------------------------------------
    has_debye = debye_spec is not None
    has_lorentz = lorentz_spec is not None
    from rfx.boundaries.depths import distributed_electric_walls
    e_wall_faces = distributed_electric_walls(grid) - _pmc_faces_frozen
    debye, lorentz = stage_dispersion_slabs(
        materials, grid.dt, debye_spec, lorentz_spec,
        n_devices, nx_per, ghost, shd, nx=nx, high_wall=high_wall)
    del materials, pec_mask, pec_shapes
    del debye_spec, lorentz_spec
    if has_debye:
        debye_coeffs_sharded, debye_state_sharded = debye
    else:
        # Absent dispersion is never read by the E update; its carry passes
        # through unchanged. Keep ranks for shard_map, one scalar per device.
        _dz = jnp.zeros((n_devices, 1, 1), dtype=jnp.float32, device=shd)
        _dz5 = jnp.zeros((n_devices, 1, 1, 1), dtype=jnp.float32, device=shd)
        debye_coeffs_sharded = DebyeCoeffs(
            ca=jax.device_put(_dz, shd),
            cb=jax.device_put(_dz, shd),
            cc=jax.device_put(_dz5, shd),
            alpha=jax.device_put(_dz5, shd),
            beta=jax.device_put(_dz5, shd),
        )
        debye_state_sharded = DebyeState(
            px=jax.device_put(_dz5, shd),
            py=jax.device_put(_dz5, shd),
            pz=jax.device_put(_dz5, shd),
        )

    if has_lorentz:
        lorentz_coeffs_sharded, lorentz_state_sharded = lorentz
    else:
        _lz = jnp.zeros((n_devices, 1, 1), dtype=jnp.float32, device=shd)
        _lz5 = jnp.zeros((n_devices, 1, 1, 1), dtype=jnp.float32, device=shd)
        lorentz_coeffs_sharded = LorentzCoeffs(
            ca=jax.device_put(_lz, shd),
            cb=jax.device_put(_lz, shd),
            cc=jax.device_put(_lz, shd),
            a=jax.device_put(_lz5, shd),
            b=jax.device_put(_lz5, shd),
            c=jax.device_put(_lz5, shd),
        )
        lorentz_state_sharded = LorentzState(
            px=jax.device_put(_lz5, shd),
            py=jax.device_put(_lz5, shd),
            pz=jax.device_put(_lz5, shd),
            px_prev=jax.device_put(_lz5, shd),
            py_prev=jax.device_put(_lz5, shd),
            pz_prev=jax.device_put(_lz5, shd),
        )

    del debye, lorentz

    # ------------------------------------------------------------------
    # CPML state
    # ------------------------------------------------------------------
    if use_cpml:
        cpml_params, cpml_state_sharded = _init_cpml_sharded(
            grid, nx_local, n_devices, mesh)
    else:
        _, cpml_state_sharded = init_cpml_slabs(grid, nx_local, n_devices, mesh=mesh)
        cpml_params = None

    # ------------------------------------------------------------------
    # Static metadata
    # ------------------------------------------------------------------
    n_src = len(sources)
    n_prb = len(probes)
    from rfx.runners._distributed_records import SlabRecords
    from rfx.stepping.slab_uniform_loop import run_uniform_slab_loop
    final_carry, probe_ts, final_state_sharded, records = run_uniform_slab_loop(
        SlabRecords=SlabRecords,
        _pmc_faces_frozen=_pmc_faces_frozen,
        _pre_injection_probes=_pre_injection_probes,
        _record_vref=_record_vref,
        cpml_params=cpml_params,
        cpml_state_sharded=cpml_state_sharded,
        curl_boundary=curl_boundary,
        debye_coeffs_sharded=debye_coeffs_sharded,
        debye_state_sharded=debye_state_sharded,
        dt=dt,
        dx=dx,
        e_wall_faces=e_wall_faces,
        exchange_interval=exchange_interval,
        ghost=ghost,
        grid=grid,
        has_debye=has_debye,
        has_lorentz=has_lorentz,
        high_wall=high_wall,
        layout=layout,
        lorentz_coeffs_sharded=lorentz_coeffs_sharded,
        lorentz_state_sharded=lorentz_state_sharded,
        mesh=mesh,
        n_cpml=n_cpml,
        n_devices=n_devices,
        n_prb=n_prb,
        n_src=n_src,
        n_steps=n_steps,
        nx=nx,
        nx_local=nx_local,
        nx_per=nx_per,
        pad_x=pad_x,
        prb_device_ids=prb_device_ids,
        prb_local_specs=prb_local_specs,
        rep=rep,
        sharded_materials=sharded_materials,
        sharded_pec_mask=sharded_pec_mask,
        sharded_state=sharded_state,
        sim=sim,
        snapshot=snapshot,
        src_device_ids=src_device_ids,
        src_local_specs=src_local_specs,
        src_waveforms=src_waveforms,
        use_cpml=use_cpml,
    )

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
        # JAX-friendly) keeps the gather in the trace. Mirrors the
        # already-correct ``distributed_nu.py::_unstack_and_gather`` -- as of
        # #1038 leg 2b it IS that function, shared.
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
    if n_prb > 0 or _pre_injection_probes:
        time_series = jnp.array(probe_ts)  # (n_steps, n_probes)
    else:
        time_series = jnp.zeros((n_steps, 0), dtype=jnp.float32)

    record_result = records.assemble(final_carry["records"], multi_process=multi_process)

    return Result(
        diagnostics=diagnostics,
        realized_geometry=geometry_record,
        **record_result,
        state=final_state,
        time_series=time_series,
        s_params=None,
        freqs=None,
        grid=grid,
        dt=grid.dt,
        freq_range=(sim._freq_max / 10, sim._freq_max, sim._boundary),
    )
