"""Run dispatch and setup after public admission."""
from __future__ import annotations

import functools

import jax.numpy as jnp

from rfx import _realized
from rfx.api._spec import Result, _warn_if_nonfinite_result
from rfx.model import conductors as _conductors
from rfx.preflight.line_stub import line_stub_admission as _line_stub_admit
from rfx.ringdown import RingdownSpec
from rfx.runners._admission import refuse_coaxial_port
from rfx.runners._exchange_interval import validate_exchange_interval
from rfx.runners.nonuniform import bind_solve_call
from rfx.simulation import SnapshotSpec
from rfx.stepping import steps_for_periods


def run_after_admission(
    self,
    *,
    n_steps: int | None = None,
    num_periods: float = 20.0,
    until_decay: float | None = None,
    decay_check_interval: int = 50,
    decay_min_steps: int = 100,
    decay_max_steps: int = 50_000,
    decay_energy_consecutive: int = 2,
    radiated_flux_box: tuple | None = None,
    flux_env_checks: int = 4,
    decay_monitor_component: str = "ez",
    decay_monitor_position: tuple[float, float, float] | None = None,
    checkpoint: bool = False,
    compute_s_params: bool | None = None,
    s_param_freqs: jnp.ndarray | None = None,
    s_param_n_steps: int | None = None,
    snapshot: SnapshotSpec | None = None,
    subpixel_smoothing: bool | str = False,
    conformal_pec: bool | None = None,
    conformal_min_weight: float = 0.1,
    devices: list | None = None,
    exchange_interval: int = 1,
    skip_preflight: bool = False,
    report_every: int | None = None,
    report_label: str = "",
    ringdown: RingdownSpec | None = None,
    until_identified: bool = False,
) -> Result:
    validate_exchange_interval(exchange_interval)
    fixed_num_periods = n_steps is None
    if until_decay is not None and self._tfsf is not None:
        from rfx.sources.sources import CustomWaveform
        if isinstance(self._tfsf.waveform, CustomWaveform):
            raise NotImplementedError(
                "TFSF CustomWaveform has no source-off time for until_decay; use n_steps")
    if until_identified is not False:
        from rfx.ringdown import refuse_until_identified
        until_identified = refuse_until_identified(
            self, ringdown, until_identified, until_decay=until_decay,
            snapshot=snapshot)
    if ringdown is not None:
        from rfx.ringdown import refuse_run_request
        refuse_run_request(self, ringdown, devices=devices,
                           until_decay=until_decay,
                           compute_s_params=compute_s_params)

    # Behaviour-neutral decay-parameter sanity advisories (post-#392
    # review): single run()-level site, before dispatch, so both lanes
    # that honour until_decay (uniform + absorbing-boundary
    # non-uniform) are covered by one check.
    self._warn_decay_param_sanity(
        until_decay=until_decay,
        decay_min_steps=decay_min_steps,
        decay_max_steps=decay_max_steps,
    )

    # ---- Stage 1 conformal PEC auto-routing ----
    # When the user passes ``conformal_pec=None`` (default), derive
    # it from ``BoundarySpec.conformal_faces()``: any axis declared
    # ``Boundary(conformal=True)`` flips conformal_pec on. Explicit
    # True/False from the caller is preserved as a power-user
    # override (e.g. for A/B regression diagnosis).
    if conformal_pec is None:
        conformal_pec = bool(self._boundary_spec.conformal_faces())

    refuse_coaxial_port(self, "run")
    compute_s_params = self._validate_run_sparameter_request(
        compute_s_params=compute_s_params,
        s_param_freqs=s_param_freqs,
        s_param_n_steps=s_param_n_steps,
        devices=devices,
    )
    _line_stub_scope = _line_stub_admit(self, s_param_freqs)

    _distributed_run = devices is not None and len(devices) > 1
    _solve_assembly = self._auto_preflight(skip=skip_preflight, context="run", check_ntff="advisory", prepare=True, distributed=_distributed_run, conformal=bool(conformal_pec))
    _diagnostics = _solve_assembly.diagnostics

    # ---- (2,4) stencil fence: reject order=4 on unsupported lanes ----
    self._check_stencil_order_supported(distributed=_distributed_run)

    # ---- W6.3 unified lane dispatch: one place decides + rejects ----
    # _dispatch_plan computes the is_nonuniform/_nu_profile boolean, runs
    # the distributed adi/upml ValueErrors + the distributed+NU grading
    # guardrail, selects the lane, and resolves n_steps for the
    # NU/distributed lanes (the uniform/adi/subgridded lanes resolve it
    # below from the grid they build and reuse).
    plan = self._dispatch_plan(
        mode="run", subpixel_smoothing=subpixel_smoothing,
        diagnostics=_diagnostics,
        n_steps=n_steps,
        num_periods=num_periods,
        devices=devices,
        exchange_interval=exchange_interval,
    )
    n_steps = plan.n_steps
    if plan.lane == "run_subgridded":
        from rfx.subgridding._notice import require_experimental
        require_experimental(
            self,
            diagnostics=_diagnostics,
        )
    if _realized.ACTIVE is not None:
        _realized.enter(self, plan.lane)
    if ringdown is not None:
        from rfx.ringdown import refuse_run_lane
        refuse_run_lane(plan.lane, sum(
            1 for p in self._ports
            if float(p.impedance) > 0.0 and p.extent is not None))

    # ---- Distributed multi-device lane ----
    if plan.lane in ("run_distributed", "run_distributed_nu"):
        from rfx.runners._distributed_entry import execute_distributed
        return execute_distributed(
            self, plan=plan, _solve_assembly=_solve_assembly,
            _diagnostics=_diagnostics,
            n_steps=n_steps, num_periods=num_periods, devices=devices,
            exchange_interval=exchange_interval, skip_preflight=skip_preflight,
            compute_s_params=compute_s_params, s_param_freqs=s_param_freqs,
            s_param_n_steps=s_param_n_steps, conformal_pec=conformal_pec,
            subpixel_smoothing=subpixel_smoothing, checkpoint=checkpoint,
            snapshot=snapshot, until_decay=until_decay, report_every=report_every)

    # ---- Non-uniform mesh lane ----
    if plan.lane == "run_nonuniform":
        # #383: absorbing NU domains support the interior-energy decay
        # stop; closed/PEC domains refuse it because their energy does
        # not decay and this lane has no point-field fallback. A single
        # port reads S from the main record; more ports run their own
        # drives of that record's length: n_steps, or the decay stop
        # under until_decay. A different s_param_n_steps is refused;
        # with until_decay, an explicit length is always refused.
        _nu_refused = {
            "snapshot": snapshot,
            "conformal_pec": bool(
                conformal_pec and self._has_pec_to_conform()),
            "s_param_n_steps": self._s_param_n_steps_off_record(
                s_param_n_steps, n_steps, until_decay),
        }
        if self._boundary not in ("cpml", "upml"):
            _nu_refused["until_decay"] = until_decay
        _uniform = "run on a uniform mesh (no dx/dy/dz profile)"
        self._refuse_unsupported_run_kwargs(
            "non-uniform mesh", _nu_refused,
            # One wire port reads S from the main record on the uniform
            # lane too; only a multi-port S-matrix runs its own records.
            instead={"*": _uniform, "s_param_n_steps": (
                _uniform if len(self._port_sparameter_entries()) > 1
                else None)},
            reason_overrides={
                "until_decay":
                    "the interior-energy decay stop needs absorbing "
                    "boundaries (cpml/upml); a closed non-uniform "
                    "domain's energy does not decay, and the "
                    "non-uniform lane has no point-field fallback"
                    "",
                "s_param_n_steps":
                    "this lane matches the S-parameter record to the main "
                    "run's port record, whose length is n_steps, or the decay stop under until_decay",
            },
        )
        if until_decay is not None:
            # #388 DC-floor predictor for the NU decay stop. The grid
            # build is pure (no sim-state mutation) and cheap; an
            # advisory must never block the run, so any build failure
            # simply skips the check (the runner will surface it).
            try:
                _nu_dt_for_dc = self._build_nonuniform_grid().dt
            except Exception:
                _nu_dt_for_dc = None
            if _nu_dt_for_dc is not None:
                from rfx.measurement import warn_dc_floor
                warn_dc_floor(self, dt=_nu_dt_for_dc, n_table=decay_max_steps)
        _nu_call = functools.partial(
            self._run_nonuniform,
            n_steps=n_steps,
            conformal_pec=conformal_pec,
            report_every=report_every,
            report_label=report_label,
            compute_s_params=compute_s_params,
            s_param_freqs=s_param_freqs,
            subpixel_smoothing=subpixel_smoothing,
            checkpoint=checkpoint,
            until_decay=until_decay,
            decay_check_interval=decay_check_interval,
            decay_min_steps=decay_min_steps,
            decay_max_steps=decay_max_steps,
            decay_energy_consecutive=decay_energy_consecutive,
            radiated_flux_box=radiated_flux_box,
            flux_env_checks=flux_env_checks,
        )
        _nu_call, _nu_grid = bind_solve_call(_solve_assembly, _nu_call)
        if ringdown is None:
            _res = _nu_call()
        elif until_identified:
            from rfx.ringdown import RingdownStop
            _res = RingdownStop(
                self, ringdown, lane="graded", n_max=n_steps,
                grid=_nu_grid).run(_nu_call)
            n_steps = _res.ringdown.stop.n_stop
        else:
            from rfx.ringdown import RingdownRun
            _res = RingdownRun(
                self, ringdown, lane="graded", n_steps=n_steps,
                grid=_nu_grid, bins=s_param_freqs).run(_nu_call)
        self._warn_run_sparams_if_nonpassive(_res)
        self._warn_postrun_energy_witness(
            _res,
            fixed_num_periods=fixed_num_periods,
            until_decay=until_decay,
        )
        _res = self._attach_run_settling_witness(
            _res, n_steps=n_steps, num_periods=num_periods)
        _warn_if_nonfinite_result(_res, context="run")
        from rfx.current_moments import require_accumulated_current_moments
        require_accumulated_current_moments(self, _res, "run")
        return _res

    conductors, grid, assembly_inputs, _run_sheet_specs, _run_pec_sheets, _run_pec_wires = _conductors.uniform_solve_inputs(_solve_assembly)
    base_materials, debye_spec, lorentz_spec, pec_mask, pec_shapes, _, kerr_chi3 = assembly_inputs
    from rfx.realized_geometry import attach_record, early_run_record
    geometry_record = early_run_record(self, conductors, plan.lane)

    if plan.lane == "run_adi":
        from rfx.materials.thin_conductor import refuse_f0_sheets
        refuse_f0_sheets(self._thin_conductors, "ADI run()")
        from rfx.current_moments import refuse_current_moment_monitor
        refuse_current_moment_monitor(self, "ADI lane (solver='adi')")
        self._refuse_unsupported_run_kwargs("ADI (solver='adi')", {
            "subpixel_smoothing": subpixel_smoothing,
            "checkpoint": checkpoint,
            "snapshot": snapshot,
            "until_decay": until_decay,
            "conformal_pec": bool(conformal_pec and pec_shapes),
            **({} if report_every is None else {"report_every": report_every}),
        }, instead="use the default solver='yee'")
        if n_steps is None:
            dt_adi = float(grid.dt * self._adi_cfl_factor)
            n_steps = steps_for_periods(dt_adi, grid.freq_max, num_periods)
        _res = self._run_adi_from_materials(
            grid,
            base_materials,
            debye_spec,
            lorentz_spec,
            n_steps=n_steps,
            pec_mask=pec_mask,
            pec_sheets=tuple(_run_pec_sheets),
            pec_wires=tuple(_run_pec_wires),
            return_state=True,
            lane="run_adi",
            diagnostics=_diagnostics,
            conformal_pec=conformal_pec,
        )
        _res = self._attach_run_settling_witness(_res, n_steps=n_steps)
        _warn_if_nonfinite_result(_res, context="run")
        from rfx.current_moments import require_accumulated_current_moments
        require_accumulated_current_moments(self, _res, "run")
        return attach_record(_res, geometry_record)

    # ---- Subgridded lane ----
    if plan.lane == "run_subgridded":
        from rfx.materials.thin_conductor import refuse_f0_sheets
        refuse_f0_sheets(self._thin_conductors, "subgridded (SBP-SAT) run()")
        self._refuse_unsupported_run_kwargs("subgridded (SBP-SAT)", {
            "subpixel_smoothing": subpixel_smoothing,
            "checkpoint": checkpoint,
            "snapshot": snapshot,
            "until_decay": until_decay,
            "conformal_pec": bool(conformal_pec and pec_shapes),
            **({} if report_every is None else {"report_every": report_every}),
        }, instead="drop add_refinement() to run on the uniform lane")
        subgrid_n_steps = n_steps
        if subgrid_n_steps is None:
            # The subgrid runner advances with the fine-grid CFL timestep
            # (dx_coarse / ratio), while ``grid.num_timesteps`` is based
            # on the coarse-grid timestep.  Preserve the user's requested
            # physical duration by scaling the automatically computed
            # coarse step count by the refinement ratio.  Explicit
            # ``n_steps`` remains a low-level escape hatch and is passed
            # through unchanged.
            subgrid_n_steps = grid.num_timesteps(num_periods=num_periods) * int(
                self._refinement["ratio"]
            )
        if _run_pec_sheets or _run_pec_wires:
            raise NotImplementedError(
                "solver='subgridded' (SBP-SAT) does not realize PEC "
                "sheets or wires: the coarse/fine SBP-SAT "
                "operators apply PEC from a cell mask on two grids "
                "and a sheet owns no cell, so a declared sheet would "
                "be silently absent. Draw the conductor as a volume "
                "(a Box at least one cell thick) or use the uniform "
                "or non-uniform lane.")
        _res = self._run_subgridded(
            grid,
            base_materials,
            pec_mask,
            n_steps=subgrid_n_steps,
            diagnostics=_diagnostics,
            conformal_pec=conformal_pec,
            compute_s_params=compute_s_params,
            s_param_freqs=s_param_freqs,
            s_param_n_steps=s_param_n_steps,
        )
        _res = self._attach_run_settling_witness(
            _res, n_steps=subgrid_n_steps)
        _warn_if_nonfinite_result(_res, context="run")
        from rfx.current_moments import require_accumulated_current_moments
        require_accumulated_current_moments(self, _res, "run")
        return attach_record(_res, geometry_record)

    # ---- Uniform path ----
    if n_steps is None:
        n_steps = grid.num_timesteps(num_periods=num_periods)

    if until_decay is not None:
        # #388 DC-floor predictor for the decay stop (rfx.measurement.decay).
        from rfx.measurement import warn_dc_floor
        warn_dc_floor(self, dt=grid.dt, n_table=decay_max_steps)

    from rfx.runners.uniform import run_uniform
    _field_dtype = self._resolve_field_dtype()
    _uniform_call = functools.partial(
        run_uniform,
        self,
        conductors=conductors,
        diagnostics=_diagnostics,
        n_steps=n_steps,
        until_decay=until_decay,
        decay_check_interval=decay_check_interval,
        decay_min_steps=decay_min_steps,
        decay_max_steps=decay_max_steps,
        decay_energy_consecutive=decay_energy_consecutive,
        radiated_flux_box=radiated_flux_box,
        flux_env_checks=flux_env_checks,
        decay_monitor_component=decay_monitor_component,
        decay_monitor_position=decay_monitor_position,
        checkpoint=checkpoint,
        compute_s_params=compute_s_params,
        s_param_freqs=s_param_freqs,
        s_param_n_steps=s_param_n_steps,
        snapshot=snapshot,
        subpixel_smoothing=subpixel_smoothing,
        conformal_pec=conformal_pec,
        conformal_min_weight=conformal_min_weight,
        pec_shapes=pec_shapes,
        grid=grid,
        base_materials=base_materials,
        debye_spec=debye_spec,
        lorentz_spec=lorentz_spec,
        pec_mask=pec_mask,
        pec_sheets=tuple(_run_pec_sheets),
        pec_wires=tuple(_run_pec_wires),
        kerr_chi3=kerr_chi3,
        field_dtype=_field_dtype,
        sheet_specs=_run_sheet_specs,
        **(
            {} if report_every is None else {"report_every": report_every, "report_label": report_label}
        ),
    )
    if ringdown is None:
        _res = _uniform_call()
    elif until_identified:
        from rfx.ringdown import RingdownStop
        _res = RingdownStop(self, ringdown, lane="uniform", n_max=n_steps,
                            grid=grid).run(_uniform_call)
        n_steps = _res.ringdown.stop.n_stop
    else:
        from rfx.ringdown import RingdownRun
        _res = RingdownRun(self, ringdown, lane="uniform", n_steps=n_steps,
                           grid=grid, bins=s_param_freqs).run(_uniform_call)
    self._warn_run_sparams_if_nonpassive(_res)
    self._warn_postrun_energy_witness(
        _res,
        fixed_num_periods=fixed_num_periods,
        until_decay=until_decay,
    )
    _res = self._attach_run_settling_witness(
        _res, n_steps=n_steps, num_periods=num_periods)
    _warn_if_nonfinite_result(_res, context="run")
    from rfx.current_moments import require_accumulated_current_moments
    require_accumulated_current_moments(self, _res, "run")
    return _res
