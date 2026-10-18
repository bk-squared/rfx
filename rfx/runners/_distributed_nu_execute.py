"""Shared graded distributed assembly, staging and forward packaging."""
from __future__ import annotations
import math
import jax
import jax.numpy as jnp
import numpy as np
from rfx import _realized, _grid_metric
from rfx.model import conductors as _conductors
from rfx.model.overrides import apply_material_overrides
from rfx._diagnostic_transport import merge_diagnostics
from rfx.api._spec import ForwardResult

def execute_from_materials(
    self,
    *, lane: str = "fwd_distributed_nu", subpixel_smoothing=False,
    diagnostics=(),
    assembly=None,
    eps_override: jnp.ndarray | None = None,
    sigma_override: jnp.ndarray | None = None,
    pec_mask_override: jnp.ndarray | None = None,
    pec_occupancy_override: jnp.ndarray | None = None,
    n_steps: int,
    checkpoint: bool = True,
    emit_time_series: bool = True,
    checkpoint_every: int | None = None,
    n_warmup: int = 0,
    devices: list | None = None,
    exchange_interval: int = 1,
    skip_preflight: bool = False,
    gather_final_state: bool = False,
    s_param_freqs=None,
    compute_s_params=False,
):
    """Stage and execute the graded distributed solve shared by run/forward.

    Run requests gathered fields and an assembly record; forward keeps the
    fields sharded. Sources, materials, ADE, CPML and PEC staging are shared.
    """
    from rfx.runners._distributed_entry import refuse_distributed_options
    refuse_distributed_options(self, subpixel_smoothing=subpixel_smoothing)
    from rfx.runners.distributed_v2 import refuse_distributed_periodic
    refuse_distributed_periodic(self, lane="distributed non-uniform forward()")
    if self._interface_eps == "dual_average":
        raise ValueError("interface_eps='dual_average' is not supported on the distributed NU lane")
    # Defense-in-depth: the distributed-NU runner does not honour
    # stencil_order (both distributed and non-uniform are unsupported).
    self._check_stencil_order_supported(distributed=True)
    if self._flux_monitors:
        raise NotImplementedError(
            "add_flux_monitor() is not supported on the distributed "
            "non-uniform forward path; the sharded NU scan body does "
            "not accumulate flux DFTs. Drop flux monitors (use "
            "add_ntff_box() for far-field observables) or use the "
            "uniform lane."
        )
    if self._dft_planes:
        raise NotImplementedError(
            "add_dft_plane_probe() is not supported on the distributed "
            "non-uniform forward path (issue #579); the sharded NU scan "
            "body does not accumulate DFT-plane fields. Drop DFT plane "
            "probes or use the uniform lane."
        )
    from rfx.current_moments import refuse_current_moment_monitor
    refuse_current_moment_monitor(
        self, "distributed non-uniform forward path")
    import warnings as _w
    from rfx.runners.distributed_nu import (
        build_sharded_nu_grid,
        init_cpml_for_sharded_nu,
        run_nonuniform_distributed_pec,
        is_forward_sharded_override,
        stage_forward_dispersion_x_slab,
    )
    from rfx.core.yee import MaterialArrays
    from rfx.nonuniform import (
        make_current_source as _nu_make_current_source,
        current_source_samples as _nu_current_source_samples,
        current_source_volume as _nu_current_source_volume,
    )
    from rfx.api._source_semantics import needs_scale as _needs_scale
    from rfx.simulation import ProbeSpec, SourceSpec

    # ---- Resolve devices (V3 §5 semantics) ----
    if devices is None:
        devices = list(jax.devices())
    else:
        available = list(jax.devices())
        # Reject lists longer than the available device count.  This
        # catches both "more devices than exist" and the duplicate-
        # entry case (a 4-element list built from 2 real devices)
        # before JAX errors out deep inside ``device_put`` with an
        # opaque ``safe_zip`` traceback.
        if len(devices) > len(available):
            raise ValueError(
                f"forward(distributed=True, devices=...): requested "
                f"{len(devices)} devices but only {len(available)} are "
                "available in jax.devices()."
            )
        # Reject duplicate entries — jax.sharding.Mesh requires each
        # device to appear at most once.
        if len(set(map(id, devices))) != len(devices):
            raise ValueError(
                "forward(distributed=True, devices=...): duplicate "
                "device entries are not allowed; each device must "
                "appear at most once."
            )
        for d in devices:
            if d not in available:
                raise ValueError(
                    f"forward(distributed=True, devices=...): device "
                    f"{d!r} is not in jax.devices() "
                    f"(available={len(available)} devices)."
                )
    n_devices = len(devices)

    # ---- Build the NU grid up front for preflight metrics ----
    grid = self._build_nonuniform_grid()

    # ---- V3 §M4 distributed-specific preflight (5 checks).  Skipped
    # entirely when the caller requested skip_preflight=True. ----
    if not skip_preflight:
        # Check 1 — device count.
        if n_devices < 2:
            raise ValueError(
                f"forward(distributed=True) requires at least 2 "
                f"devices; found {n_devices} (devices={devices!r}). "
                "Use distributed=False on a single device, or pass an "
                "explicit devices list with len>=2."
            )

        # Check 2 — grading ratio (hard error at >5).  Use the same
        # max-over-all-axes definition as Simulation.run().
        _max_ratio = 1.0
        for _prof in (
            self._dx_profile, self._dy_profile, self._dz_profile,
        ):
            if _prof is not None and len(_prof) > 0:
                _pa = np.asarray(_prof, dtype=np.float64)
                if float(_pa.min()) > 0.0:
                    _max_ratio = max(
                        _max_ratio,
                        float(_pa.max()) / float(_pa.min()),
                    )
        if _max_ratio > 5.0:
            raise ValueError(
                f"grading_ratio={_max_ratio:.2f} exceeds 5.0; "
                "distributed NU forward requires grading_ratio <= 5.0 "
                "for shared-dt stability and x-face CPML calibration. "
                "Reduce the cell-size variation or omit distributed=True."
            )

        # Check 3 — ghost width vs local slab.  Mirror
        # ``ShardedNUGrid`` arithmetic for nx_per_rank.
        nx = grid.nx
        pad_x = 0
        if nx % n_devices != 0:
            pad_x = n_devices - (nx % n_devices)
        nx_padded = nx + pad_x
        nx_per_rank = nx_padded // n_devices
        ghost_width = math.floor(exchange_interval / 2) + 1
        for rank in range(n_devices):
            if ghost_width > nx_per_rank:
                raise ValueError(
                    f"ghost_width={ghost_width} exceeds "
                    f"nx_per_rank={nx_per_rank} for rank {rank}; "
                    "reduce exchange_interval or increase nx."
                )

        # CPML width is an unconditional admission check in the shared
        # initializer, using each active face and the last rank's padding.

        # Check 5 — segmented remat overhead warning.
        if (
            n_warmup == 0
            and checkpoint_every is not None
            and n_steps < 1000
        ):
            _w.warn(
                f"checkpoint_every={checkpoint_every} with "
                f"n_warmup=0 and n_steps={n_steps} < 1000 may spend "
                "more time on recomputation overhead than it saves "
                "in memory; consider checkpoint_every=None for small "
                "runs.",
                UserWarning,
                stacklevel=3,
            )

    sharded_grid = build_sharded_nu_grid(
        grid, n_devices, exchange_interval=exchange_interval,
    )
    from jax.sharding import Mesh
    mesh = Mesh(np.array(devices), axis_names=("x",))
    multiprocess = any(d.process_index != jax.process_index() for d in devices)
    for name, override in (("eps_override", eps_override),
                           ("sigma_override", sigma_override),
                           ("pec_occupancy_override", pec_occupancy_override)):
        if override is not None:
            sharded = is_forward_sharded_override(override, sharded_grid, mesh)
            if not sharded and tuple(override.shape) != tuple(grid.shape):
                # A padded-layout array in any form but the x-sharded one
                # would feed its pad row in as a real cell (eps 0 there
                # gives a NaN gradient); only the grid shape is local.
                raise ValueError(
                    f"{name} has shape {tuple(override.shape)}: a local override "
                    f"must have the grid shape {tuple(grid.shape)}; the padded "
                    f"shape {(sharded_grid.nx_padded, sharded_grid.ny, sharded_grid.nz)} "
                    "is accepted only as an x-sharded global array with the "
                    "sharding from Simulation.distributed_override_layout(devices).")
            if multiprocess and not sharded:
                raise ValueError(
                    f"{name} across processes requires an x-sharded global array "
                    "with NamedSharding(mesh, P('x')) and the shape returned by "
                    "Simulation.distributed_override_layout(devices). "
                    "Local whole-domain overrides are supported only in one process.")

    # ---- Assemble full-domain materials ----
    (
        conductors,
        (materials, debye_spec, lorentz_spec, pec_mask),
        _dnu_pec_sheets,
        _dnu_pec_wires,
        solve_diagnostics,
    ) = _conductors.distributed_solve_inputs(self, grid, assembly, skip_preflight, gather_final_state)
    diagnostics = merge_diagnostics(diagnostics, solve_diagnostics)
    from rfx.model.distributed_ports import assemble
    materials, pec_mask, pec_edges, geometry_record, sources, material_drive, port_specs, pec_occupancy_override = assemble(
        self, grid, materials, conductors, pec_mask, _dnu_pec_sheets, _dnu_pec_wires,
        eps_override=eps_override, sigma_override=sigma_override, pec_mask_override=pec_mask_override,
        pec_occupancy_override=pec_occupancy_override, debye_spec=debye_spec, lorentz_spec=lorentz_spec, n_steps=n_steps, lane=lane,
        gather=gather_final_state, diagnostics=diagnostics, s_param_freqs=s_param_freqs)
    del conductors
    from rfx.stepping.slab_ports import global_plan
    port_plan = global_plan(self, grid, port_specs, lane, s_param_freqs, compute_s_params)

    # Stage one input at a time. The source normalization above has
    # consumed concrete cell scalars (a material-driven source reads its
    # Cb from the staged slabs in the runner instead); no second
    # MaterialArrays may keep the original whole-domain eps/sigma alive
    # during the scan.
    if _realized.ACTIVE is not None:
        _realized.sources(grid, materials, sources, "distributed_nu.sources")
    from rfx.model.electric_metrics import stage_forward_materials
    sharded_grid = sharded_grid.select_step(debye_spec, lorentz_spec, self._boundary_spec, ports=bool(port_specs))
    sharded_materials = stage_forward_materials(materials, sharded_grid.layout, mesh, uniform=sharded_grid.step_kind == 'wrapped')
    del materials

    sharded_pec_mask = _conductors.stage_distributed_edges(
        pec_edges, sharded_grid, mesh, multiprocess,
        inputs=None if port_specs else (pec_mask, _dnu_pec_sheets, _dnu_pec_wires))
    del pec_mask, pec_edges
    sharded_pec_occupancy = _conductors.stage_distributed_occupancy(pec_occupancy_override, sharded_grid, mesh)
    del pec_occupancy_override

    sharded_debye = stage_forward_dispersion_x_slab(
        sharded_materials, grid.dt, debye_spec, sharded_grid, mesh, "debye",
    )
    del debye_spec
    sharded_lorentz = stage_forward_dispersion_x_slab(
        sharded_materials, grid.dt, lorentz_spec, sharded_grid, mesh, "lorentz",
    )
    del lorentz_spec

    # ---- CPML init + sharding (Phase 2C). ----
    cpml_params = None
    cpml_state_sharded = None
    cpml_layers = int(getattr(self, "_cpml_layers", 0) or 0)
    if self._boundary == "cpml" and cpml_layers > 0:
        cpml_params, cpml_state_sharded = init_cpml_for_sharded_nu(
            sharded_grid, n_devices, mesh=mesh,
            pec_faces=getattr(self, "_pec_faces", None),
            pmc_faces=self._boundary_spec.pmc_faces(),
        )

    probes: list[ProbeSpec] = []
    for pe in self._probes:
        idx = _grid_metric.field_index(grid, pe.position, pe.component)
        probes.append(ProbeSpec(
            i=int(idx[0]), j=int(idx[1]), k=int(idx[2]),
            component=pe.component,
        ))

    # ---- Launch the sharded NU runner. ----
    result = run_nonuniform_distributed_pec(
        sharded_grid,
        sharded_materials,
        sharded_pec_mask,
        n_steps,
        sources=sources,
        probes=probes,
        n_devices=n_devices,
        exchange_interval=exchange_interval,
        debye=sharded_debye,
        lorentz=sharded_lorentz,
        devices=devices,
        cpml_params=cpml_params,
        cpml_state=cpml_state_sharded,
        sharded_pec_occupancy=sharded_pec_occupancy,
        checkpoint_every=checkpoint_every,
        n_warmup=n_warmup,
        emit_time_series=emit_time_series,
        gather_final_state=gather_final_state,
        pmc_faces=frozenset(self._boundary_spec.pmc_faces()),
        material_drive=tuple(material_drive),
        port_plan=port_plan,
        prepare_only=port_plan is not None,
    )
    if port_plan is not None:
        del sharded_materials
        result = result()

    return (
        grid,
        dict(
            result,
            diagnostics=diagnostics,
        ),
        geometry_record,
    )

def forward_from_materials(
    self, *, diagnostics=(), **kwargs
) -> ForwardResult:
    """Pack the shared distributed graded solve for differentiable callers."""
    grid, result, _ = self._execute_distributed_nonuniform_from_materials(
        diagnostics=diagnostics,
        **kwargs,
    )
    from rfx.api._forward_result import pack_nu_forward_result
    return pack_nu_forward_result(
        diagnostics=result.get('diagnostics', diagnostics),
        time_series=result.get('time_series'), grid=grid, dt=grid.dt,
        s_params=result.get('s_params'), freqs=result.get('s_param_freqs'),
        wire_port_sparams=result.get('wire_port_sparams'),
        lumped_port_sparams=result.get('lumped_port_sparams'),
        sparam_time_records=result.get('sparam_time_records'),
        sparam_read_ids=result.get('sparam_read_ids'),
        dft_planes=result.get('dft_planes'))


def dispatch_forward(self, *, plan, _solve_assembly, subpixel_smoothing=False, eps_override, sigma_override, pec_mask_override, pec_occupancy_override, checkpoint, emit_time_series, checkpoint_every, n_warmup, devices, exchange_interval, skip_preflight, port_s11_freqs, num_periods, design_sigma_override, design_eps_override):
    from rfx.api._gradient_witness import with_conductivity_record
    from rfx.materials.thin_conductor import refuse_f0_sheets
    refuse_f0_sheets(self._thin_conductors,
                     "distributed non-uniform forward()")
    result = self._forward_distributed_nonuniform_from_materials(
        subpixel_smoothing=subpixel_smoothing,
        diagnostics=_solve_assembly.diagnostics,
        assembly=_solve_assembly,
        eps_override=eps_override,
        sigma_override=sigma_override,
        pec_mask_override=pec_mask_override,
        pec_occupancy_override=pec_occupancy_override,
        n_steps=plan.n_steps,
        checkpoint=checkpoint,
        emit_time_series=emit_time_series,
        checkpoint_every=checkpoint_every,
        n_warmup=n_warmup,
        devices=devices,
        exchange_interval=exchange_interval,
        skip_preflight=skip_preflight,
        s_param_freqs=port_s11_freqs,
    )
    return with_conductivity_record(self._attach_run_settling_witness(
        result, n_steps=plan.n_steps, num_periods=num_periods, context="forward"),
        design_sigma=design_sigma_override, sigma_override=sigma_override, design_eps=design_eps_override, n_steps=plan.n_steps, dt=getattr(result.grid, "dt", None))

