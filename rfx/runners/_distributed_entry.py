"""Shared public entry seam for the distributed runners; no stepping here."""

import jax.numpy as jnp
import numpy as np

from rfx.api._spec import Result, _warn_if_nonfinite_result


def refuse_distributed_options(self, *, subpixel_smoothing=False, **options):
    """Use the same option refusal on imperative and differentiable entries."""
    # One device on a graded mesh is the non-uniform lane, which
    # carries neither snapshot nor conformal_pec, nor until_decay on
    # closed boundaries.
    _one_device = "omit devices= to run on one device"
    _dist_instead = _one_device
    if self._uses_nonuniform_mesh:
        _one_uniform = ("omit devices= and run on a uniform mesh "
                        "(no dx/dy/dz profile)")
        _dist_instead = {"*": _one_device,
                         "snapshot": _one_uniform,
                         "conformal_pec": _one_uniform}
        if self._boundary not in ("cpml", "upml"):
            _dist_instead["until_decay"] = _one_uniform
    self._refuse_unsupported_run_kwargs("distributed multi-device", {
        "subpixel_smoothing": subpixel_smoothing, **options,
    }, instead=_dist_instead)


def execute_distributed(self, *, plan, _solve_assembly, _diagnostics, _call_args,
                        n_steps, num_periods, devices, exchange_interval=1,
                        skip_preflight=False, compute_s_params=False,
                        s_param_freqs=None, s_param_n_steps=None,
                        conformal_pec=False, subpixel_smoothing=False,
                        checkpoint=False, snapshot=None, until_decay=None,
                        report_every=None, _forward_port_s11=False):
    """Use the run entry's admission, fallback, execution and S-matrix path."""
    # ---- Distributed multi-device lane ----
    if self._interface_eps == "dual_average":
        raise ValueError("interface_eps='dual_average' is not supported on the distributed lane")
    from rfx.runners.distributed_v2 import one_device_reason, run_on_one_device
    if reason := one_device_reason(self):
        del _solve_assembly  # the fallback performs its own solve assembly
        return run_on_one_device(self, {**_call_args, "devices": None,
                                 "exchange_interval": 1, "skip_preflight": True},
                                 reason=reason, requested_devices=len(devices),
                                 diagnostics=_diagnostics)
    if self._dft_planes:
        raise NotImplementedError(
            "add_dft_plane_probe() is not supported on the "
            "distributed multi-device run() path (issue #579); "
            "rfx.runners.distributed_v2 does not accumulate DFT-plane fields, "
            "so a registered plane would be silently dropped. Drop "
            "DFT plane probes or omit devices=... (use a "
            "single-device run() instead)."
        )
    from rfx.model.distributed_ports import (
        refuse_unsupported_distributed_features,
    )
    refuse_unsupported_distributed_features(
        self,
        lane="distributed multi-device run()",
        diagnostics=_diagnostics,
    )
    _dist_conformal = bool(conformal_pec and self._has_pec_to_conform())
    # Resolve the default as the one-device lane of this mesh does: the
    # uniform runner computes S for every impedance port, the non-uniform
    # runner only for wire ports (extent set).
    if compute_s_params is None:
        if self._uses_nonuniform_mesh:
            compute_s_params = any(pe.impedance != 0.0 and pe.extent is not None
                                   for pe in self._ports)
        else:
            compute_s_params = any(pe.impedance != 0.0 for pe in self._ports)
        # Use the same refusal as an explicit S-parameter request.
        self._validate_run_sparameter_request(
            compute_s_params=compute_s_params,
            s_param_freqs=s_param_freqs,
            s_param_n_steps=s_param_n_steps,
            devices=devices,
        )
    if compute_s_params:
        _dist_ports = [pe for pe in self._ports if pe.impedance != 0.0]
        if len(_dist_ports) == 1 and _dist_ports[0].extent is not None:
            self._refuse_unsupported_run_kwargs(
                "uniform single-wire-port S-parameter", {
                    "s_param_n_steps": self._s_param_n_steps_off_record(
                        s_param_n_steps, n_steps, until_decay),
                }, instead=None,
                reason_overrides={"s_param_n_steps":
                    "this lane computes S11 with a second full distributed "
                    "run with all sources on, using the main run's "
                    "n_steps and source convention"})
        from rfx.probes.sparam_driver import refuse_distributed_lumped_s_pmc
        refuse_distributed_lumped_s_pmc(self)

    refuse_distributed_options(self, subpixel_smoothing=subpixel_smoothing,
        checkpoint=checkpoint, snapshot=snapshot, until_decay=until_decay,
        conformal_pec=_dist_conformal,
        **({} if report_every is None else {"report_every": report_every}))
    from rfx.materials.thin_conductor import refuse_f0_sheets
    refuse_f0_sheets(self._thin_conductors, "distributed multi-device run()")
    from rfx.runners._admission import admit_run_s_matrix
    admit_run_s_matrix(self, compute_s_params=compute_s_params,
                       conformal_pec=conformal_pec, distributed=True)
    if plan.lane == "run_distributed_nu":
        # Same drive-isolation refusal as one-device run_nonuniform, before the solve.
        if (compute_s_params is not False
                and any(p.impedance > 0 and p.extent is not None for p in self._ports)):
            from rfx.runners._admission import require_isolated_s_drive
            require_isolated_s_drive(self, fields_only=True)
        from rfx.runners.distributed_v2 import _spans_other_processes
        if devices is not None and _spans_other_processes(devices):
            # The graded runner does not gather the final fields
            # across processes (#1461 review), so run() could not
            # return them; forward(distributed=True) can.
            raise NotImplementedError(
                "run(devices=...) on a non-uniform mesh does not support "
                "devices from more than one JAX process: the final fields "
                "are not gathered across processes. Use devices from one "
                "process, or forward(distributed=True) for the probe traces.")
        grid, result, geometry_record = self._execute_distributed_nonuniform_from_materials(
            assembly=_solve_assembly,
            n_steps=n_steps,
            devices=devices,
            exchange_interval=exchange_interval,
            skip_preflight=skip_preflight,
            gather_final_state=True,
            lane="run_distributed_nu",
            diagnostics=_diagnostics,
            s_param_freqs=s_param_freqs,
            compute_s_params=compute_s_params,
        )
        _res = Result(
            diagnostics=result.get("diagnostics", _diagnostics),
            state=result["final_state"],
            time_series=result["time_series"],
            s_params=result.get('s_params'),
            freqs=result.get('s_param_freqs'),
            wire_port_sparams=(tuple(zip(result['wire_sparams_meta'], result['wire_sparams_raw']))
                               if 'wire_sparams_raw' in result else None),
            sparam_time_records=result.get('sparam_time_records'),
            sparam_read_ids=result.get('sparam_read_ids'),
            grid=grid,
            dt=grid.dt,
            freq_range=(self._freq_max / 10, self._freq_max, self._boundary),
            realized_geometry=geometry_record,
        )
    else:
        from rfx.runners.distributed_v2 import run_distributed
        _res = run_distributed(
            self,
            n_steps=n_steps,
            devices=devices,
            diagnostics=_diagnostics,
            exchange_interval=exchange_interval,
            assembly=_solve_assembly,
            conformal_pec=conformal_pec,
        )
    if compute_s_params and plan.lane != "run_distributed_nu":
        from rfx.measurement import default_frequency_grid
        from rfx.probes.sparam_driver import compute_lumped_wire_s_matrix_via_scan
        if s_param_freqs is None:
            s_param_freqs = default_frequency_grid(self._freq_max, 50)
        _s_params, _ = compute_lumped_wire_s_matrix_via_scan(
            self, s_param_freqs,
            n_steps=s_param_n_steps if s_param_n_steps is not None else n_steps,
            devices=devices,
            **({"_main_wire_record": True} if len(_dist_ports) == 1
               and _dist_ports[0].extent is not None else {}),
            **({"_forward_port_s11": True} if _forward_port_s11 else {}),
        )
        _res = _res._replace(s_params=_s_params, freqs=np.asarray(s_param_freqs))
    _res = self._attach_run_settling_witness(
        _res, n_steps=n_steps, num_periods=num_periods)
    _warn_if_nonfinite_result(_res, context="run")
    from rfx.current_moments import require_accumulated_current_moments
    require_accumulated_current_moments(self, _res, "run")
    return _res



# What a multi-device forward() on a uniform mesh refuses, in the user's words:
# why, and what runs that request today. Formatted by the run entry's refusal.
_NO_DESIGN_VARIABLE = ("a multi-device forward() on a uniform mesh takes none of "
                       "forward()'s override or design arguments")
_ONE_DEVICE = "run on one device (omit distributed=True)"
_ON_A_GRADED_MESH = "use a graded mesh, where forward(distributed=True) takes this override"
_GRADED_TOO = ("eps_override", "sigma_override", "pec_mask_override", "pec_occupancy_override")
_ONE_DEVICE_ONLY = ("mu_r_override", "design_box", "design_eps_override", "design_sigma_override",
                    "design_occupancy_override", "design_box_holds_ports", "rlc_values_override")
_CHECKPOINTS = ("checkpoint", "checkpoint_every", "checkpoint_segments")
FORWARD_REFUSALS = {
    **{name: _NO_DESIGN_VARIABLE for name in _GRADED_TOO + _ONE_DEVICE_ONLY},
    **{name: "the multi-device time loop on a uniform mesh keeps no checkpoints, and with no "
             "design variable there is no gradient to save memory for" for name in _CHECKPOINTS},
    "n_warmup": "every step of a multi-device forward() on a uniform mesh is recorded",
    "emit_time_series": "a multi-device forward() on a uniform mesh always returns the probe records",
}
FORWARD_REMEDIES = {
    **{name: _ON_A_GRADED_MESH for name in _GRADED_TOO},
    **{name: "omit the checkpoint arguments" for name in _CHECKPOINTS},
    "n_warmup": "omit n_warmup",
    "emit_time_series": "omit emit_time_series",
}


def forward_distributed(sim, *, plan, assembly, devices, exchange_interval,
                        num_periods, port_s11_freqs, skip_preflight,
                        checkpoint, emit_time_series, n_warmup,
                        design_box_holds_ports, subpixel_smoothing=False, **overrides):
    """Admit the fixed-model forward request, execute once, and pack observables."""
    import jax
    from rfx.api._forward_result import pack_nu_forward_result
    from rfx.core.jax_utils import is_tracer

    if is_tracer(jnp.zeros(())):
        # The uniform multi-device runner reads concrete values while it runs
        # (measured: jax.jit of this call fails inside it with a tracer error).
        raise NotImplementedError(
            "forward(distributed=True) on a uniform mesh cannot run inside jax.jit "
            "(or another JAX trace that stages it): call it directly. For a staged or "
            "differentiated solve with override or design arguments, run on one device "
            "(omit distributed=True) or use a graded mesh.")
    sim._refuse_unsupported_run_kwargs(
        "uniform distributed forward", {
            **overrides,
            "checkpoint": False if checkpoint is None else checkpoint,
            "emit_time_series": None if emit_time_series else False,
            "n_warmup": n_warmup if n_warmup else None,
            "design_box_holds_ports": True if design_box_holds_ports else None,
        }, instead=_ONE_DEVICE, reason_overrides=FORWARD_REFUSALS,
        remedy_overrides=FORWARD_REMEDIES, entry="Simulation.forward()")
    devices = jax.devices() if devices is None else devices
    call_args = dict(n_steps=plan.n_steps, num_periods=num_periods,
                     subpixel_smoothing=subpixel_smoothing,
                     compute_s_params=port_s11_freqs is not None,
                     s_param_freqs=port_s11_freqs, skip_preflight=skip_preflight)
    try:
        result = execute_distributed(
            sim, plan=plan, _solve_assembly=assembly,
            _diagnostics=assembly.diagnostics, _call_args=call_args,
            devices=devices, exchange_interval=exchange_interval, _forward_port_s11=True,
            conformal_pec=bool(sim._boundary_spec.conformal_faces()), **call_args)
    except (jax.errors.TracerArrayConversionError, jax.errors.ConcretizationTypeError) as error:
        # Measured: jax.grad through port_s11_freqs fails inside the S driver,
        # which reads its records as concrete arrays.
        raise NotImplementedError(
            "forward(distributed=True) on a uniform mesh cannot differentiate this request "
            "(its port S-parameters from port_s11_freqs are measured outside the trace): "
            "take this gradient on one device (omit distributed=True).") from error
    packed = pack_nu_forward_result(
        diagnostics=result.diagnostics, time_series=result.time_series,
        grid=result.grid, dt=result.dt, dft_planes=result.dft_planes,
        flux_monitors=result.flux_monitors, waveguide_ports=result.waveguide_ports,
        waveguide_sparams=result.waveguide_sparams,
        ntff_data=result.ntff_data, ntff_box=result.ntff_box,
        s_params=result.s_params, freqs=result.freqs,
        wire_port_sparams=result.wire_port_sparams,
        sparam_time_records=result.sparam_time_records,
        dft_time_records=result.dft_time_records,
        current_moment_data=result.current_moment_data,
        current_moment_monitor=result.current_moment_monitor)
    packed = sim._attach_run_settling_witness(
        packed, n_steps=plan.n_steps, num_periods=num_periods, context="forward")
    ports = [pe for pe in sim._ports if pe.impedance != 0]
    if port_s11_freqs is not None and ports:
        from rfx.runners._uniform_forward_result import pack_port_result
        packed = pack_port_result(packed, wire=ports[0].extent is not None)
    return packed
