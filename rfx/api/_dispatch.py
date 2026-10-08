"""Shared run/forward lane admission, before execution or sharding."""
from __future__ import annotations

from typing import NamedTuple
import numpy as np


class _DispatchPlan(NamedTuple):
    """Resolved execution lane for a single ``run()`` / ``forward()`` call.

    Produced by :meth:`_ExecuteMixin._dispatch_plan`, the one place that
    selects an execution lane and rejects unsupported config combinations
    (W6.3). Both callers consume the same plan, so the lane decision and
    every ``NotImplementedError`` / ``ValueError`` lane guard live together
    instead of being duplicated across ``run()`` and ``forward()``.

    Fields
    ------
    lane:
        Lane token (see ``_dispatch_plan`` for the closed set per mode).
    n_steps:
        Resolved timestep count for lanes whose step count is derived from
        a *throwaway* grid build (the non-uniform / distributed lanes). For
        lanes that build-and-reuse a grid (uniform / adi / subgridded), this
        is ``None`` and the caller resolves ``n_steps`` from the grid it
        already holds — avoiding a duplicate grid build.

    Note
    ----
    The original roadmap sketch had a third ``resolved_dz_profile`` field.
    W1.3 (commit 7a02607) moved dz-profile synthesis into
    ``_build_nonuniform_grid()`` (pure, no sim-state mutation), so there is
    no longer any dz profile to resolve at the dispatch layer; the field is
    intentionally omitted.
    """

    lane: str
    n_steps: int | None



def _dispatch_plan(
    self,
    *,
    mode: str,
    n_steps: int | None,
    num_periods: float,
    # forward-only inputs
    tfsf_material_overrides: tuple = (),
    distributed: bool = False,
    port_s11_freqs: object | None = None,
    checkpoint_segments: int | None = None,
    emit_time_series: bool = True,
    checkpoint_every: int | None = None,
    n_warmup: int = 0,
    # run-only inputs
    devices: list | None = None,
    exchange_interval: int = 1,
) -> _DispatchPlan:
    """Select the execution lane and reject unsupported config combos.

    The single decision-and-rejection point consumed by both
    :meth:`run` and :meth:`forward` (W6.3). Shared routing guards
    (``NotImplementedError`` for unsupported combinations and
    distributed-lane ``ValueError`` guardrails) run here. Individual
    entry points and runners apply additional admission checks.
    In particular, non-uniform meshes refuse refinement and unsupported
    dimensional modes before dispatch. Uniform forward entry points also
    refuse refinement when entering their material-based solve.

    ``mode`` is ``"forward"`` or ``"run"``. The two modes share the
    ``is_nonuniform`` boolean and the NU ``n_steps`` formula but have
    disjoint lane-token sets:

    - forward: ``fwd_distributed_nu`` / ``fwd_nonuniform`` / ``fwd_uniform``
    - run: ``run_distributed`` / ``run_distributed_nu`` / ``run_nonuniform`` / ``run_adi`` /
      ``run_subgridded`` / ``run_uniform``

    ``n_steps`` is returned resolved for the NU/distributed lanes (whose
    step count comes from a throwaway grid) and ``None`` for lanes that
    build-and-reuse a grid (the caller resolves it there).
    """
    from rfx.boundaries.features import admit_waveguide
    admit_waveguide(self, lane=f"{mode} dispatch")
    from rfx.boundaries.tfsf import admit_simulation
    admit_simulation(self, material_overrides=tfsf_material_overrides)
    # Constructor checks cover explicit profiles only. Geometry can make
    # auto mesh non-uniform later, so validate the declared solver against
    # the completed mesh before ANY Yee lane (including distributed) wins
    # dispatch. This is execution legality, independent of preflight.
    if self._solver == "adi":
        self._require_uniform_mesh("solver='adi'")
    self._require_mode_the_nonuniform_lane_solves()
    # Every non-uniform lane below, single-device and distributed, run()
    # and forward(), has no subgrid and would drop a refinement (#1240).
    self._require_no_refinement_on_the_nonuniform_lane()
    # A point feature one cell off a conductor drawn at the same half
    # node (#1295, #1342); run() and forward(), both lanes.
    self._require_no_half_node_split()
    is_nonuniform = self._uses_nonuniform_mesh

    if self._tfsf is not None:
        from rfx.sources.sources import CustomWaveform
        extended_tfsf = (self._tfsf.closed_box
                         or isinstance(self._tfsf.waveform, CustomWaveform))
        if extended_tfsf and (
            is_nonuniform or distributed or (devices is not None and len(devices) > 1)
            or self._refinement is not None or self._mode != "3d"
            or self._solver != "yee" or self._stencil_order != 2
        ):
            raise NotImplementedError(
                "CustomWaveform and closed_box TFSF require a uniform, single-device, "
                "3D second-order Yee run/forward with no subgrids")
        if self._tfsf.closed_box and any(
            getattr(getattr(self._boundary_spec, axis), side) != "cpml"
            for axis in "xyz" for side in ("lo", "hi")
        ):
            raise NotImplementedError("closed_box TFSF requires CPML on all six domain faces")

    def _reject_lane_precision(lane: str) -> None:
        # Issue #630 follow-up: field_dtype is threaded ONLY on the
        # uniform-lane runner (rfx/runners/uniform.py). The non-uniform,
        # distributed, distributed-NU, and subgridded runners have zero
        # field_dtype occurrences, so a non-float32 precision would
        # silently run float32 fields there -- the exact SILENT_WRONG
        # class this repo removes on sight. distributed=True is a
        # call-time kwarg invisible to preflight (see the "P3
        # (Distributed path)" precedent in
        # _validate_cfg_subgrid_limitations), so THIS is the only
        # enforcement point for that case; for the non-uniform-mesh
        # case _validate_cfg_precision_x64 also warns in advance, but
        # this raise is what actually stops it (and cannot be bypassed
        # with skip_preflight=True, unlike a preflight warning).
        if self._precision != "float32":
            raise NotImplementedError(
                f"precision={self._precision!r} is currently supported "
                f"only on the uniform single-device lane (issue #630); "
                f"the {lane!r} lane does not thread field_dtype through "
                "its runner and would silently run float32 fields "
                "regardless of this setting. Use precision='float32' "
                "(the default) here, or drop distributed=True / the "
                "non-uniform mesh profile / refinement (subgridding) to "
                "reach the uniform lane, where this knob is honoured."
            )

    if mode == "forward":
        def _fwd_nu_n_steps() -> int:
            # Throwaway NU grid build for the step count; mirrors the
            # original forward() inline form (period = 1/freq_max).
            grid_probe = self._build_nonuniform_grid()
            period = 1.0 / float(self._freq_max)
            return int(np.ceil(
                num_periods * period / float(grid_probe.dt)))

        # Distributed runners do not carry port S-parameter accumulators.
        if port_s11_freqs is not None and distributed:
            raise NotImplementedError(
                "forward(port_s11_freqs=...) is supported only on "
                "single-device forward paths (issue #72). Drop "
                "distributed=True to request port S-parameter frequencies."
            )

        # Issue #73: forward(checkpoint_segments=...) is currently wired only
        # on the uniform single-device path. Reject loudly elsewhere — both
        # for distributed=True and for non-uniform meshes — so users don't
        # get a silent fall-back to the linear-memory scan that this kwarg
        # was meant to fix. NU follow-up will mirror the pattern in
        # run_nonuniform; track on issue #73.
        if checkpoint_segments is not None and (distributed or is_nonuniform):
            raise NotImplementedError(
                "forward(checkpoint_segments=...) is currently wired only "
                "on the uniform single-device forward path (issue #73). "
                "Drop checkpoint_segments or run on a uniform mesh without "
                "distributed=True. NU support is tracked as a follow-up."
            )

        # Phase 3: distributed dispatch (V3 lines 842-847).
        if distributed:
            # NU-only in v1.6.2 (DP3 locked decision).
            if not is_nonuniform:
                raise NotImplementedError(
                    "distributed=True on forward() is currently implemented "
                    "only for non-uniform meshes; use run(..., devices=...) "
                    "for the uniform distributed path."
                )
            # Reject TFSF / waveguide ports up front (V3 §3 unsupported).
            if self._tfsf is not None:
                raise NotImplementedError(
                    "TFSF sources are not supported on the distributed "
                    "forward path; remove the TFSF source or omit "
                    "distributed=True."
                )
            if self._waveguide_ports:
                raise NotImplementedError(
                    "Waveguide ports are not supported on the distributed "
                    "forward path; remove waveguide ports or omit "
                    "distributed=True."
                )
            # T8 (2026-04): PMC is now wired across all three sharded
            # runners (distributed_nu, distributed_v2, distributed).
            # The reject guard that used to live here (introduced in
            # f3cab7c) has been removed. The single-device PMC runtime
            # hook lives in rfx/simulation.py:703-705 and the sharded
            # PMC helpers live in each runner next to their PEC analog.
            self._reject_upml_on_nonuniform(
                "distributed non-uniform forward()")
            _n = n_steps if n_steps is not None else _fwd_nu_n_steps()
            _reject_lane_precision("fwd_distributed_nu")
            return _DispatchPlan(lane="fwd_distributed_nu", n_steps=_n)

        if is_nonuniform:
            # Let the NU runner build grid/materials so it can apply the
            # NU-aware pec_mask and port/source setup against per-axis widths.
            self._reject_upml_on_nonuniform("non-uniform forward()")
            _n = n_steps if n_steps is not None else _fwd_nu_n_steps()
            _reject_lane_precision("fwd_nonuniform")
            return _DispatchPlan(lane="fwd_nonuniform", n_steps=_n)

        # Uniform forward lane: the remaining kwargs are NU-only.
        if not emit_time_series:
            raise NotImplementedError(
                "emit_time_series=False is currently only supported on the "
                "non-uniform forward path. Frequency-domain objectives "
                "(NTFF, S-params) on uniform meshes still emit time series."
            )
        if checkpoint_every is not None:
            raise NotImplementedError(
                "checkpoint_every (segmented remat) is currently only "
                "supported on the non-uniform forward path. For the "
                "uniform path, use checkpoint_segments instead (issue #73)."
            )
        if n_warmup > 0:
            raise NotImplementedError(
                "n_warmup (issue #40) is currently only supported on the "
                "non-uniform / distributed-non-uniform forward paths "
                "(_forward_from_materials has no warmup-split parameter); "
                "on the uniform lane it was previously accepted and "
                "silently ignored (issue #626 — measured bit-identical "
                "gradient at n_warmup=0 vs n_warmup=60), which is worse "
                "than an error. For reverse-mode memory relief on the "
                "uniform lane use checkpoint_segments instead (issue #73) "
                "— it is EXACT (no gradient approximation), unlike "
                "n_warmup which trades memory/compute for a truncated, "
                "increasingly biased gradient as n_warmup approaches the "
                "loss window (measured on the non-uniform lane, issue "
                "#626 part 2)."
            )
        # n_steps for the uniform forward lane is resolved by the caller
        # from the grid it builds and reuses for material assembly.
        return _DispatchPlan(lane="fwd_uniform", n_steps=n_steps)

    # mode == "run"
    distributed_run = devices is not None and len(devices) > 1
    if self._solver == "adi" and distributed_run:
        raise ValueError("solver='adi' does not support distributed execution")
    if self._boundary == "upml" and distributed_run:
        raise ValueError("boundary='upml' does not support distributed execution")

    # Distributed graded meshes share forward's NU runner and staging.
    # Keep the run() grading/TFSF admission before either runner starts.
    if distributed_run and is_nonuniform:
        import warnings as _wmod
        # Grading ratio check (shared single dt) across provided profiles.
        _max_ratio = 1.0
        for _prof in (
            self._dx_profile, self._dy_profile, self._dz_profile
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
                "Distributed + non-uniform requires grading ratio "
                "<= 5:1 for shared-dt stability; got "
                f"{_max_ratio:.2f}:1."
            )
        if self._tfsf is not None:
            raise ValueError(
                "Distributed + non-uniform does not support TFSF "
                "plane-wave sources (Phase B scope)."
            )
        if self._solver == "adi":
            raise ValueError(
                "Distributed + non-uniform does not support solver='adi'."
            )
        if _max_ratio > 3.0:
            _wmod.warn(
                f"Distributed + non-uniform grading ratio {_max_ratio:.2f}"
                ":1 exceeds the 3:1 stability caution threshold. "
                "Monitor for numerical dispersion / late-time drift.",
                stacklevel=2,
            )

    # ---- Distributed multi-device lane ----
    if distributed_run:
        _n = n_steps
        if _n is None:
            if is_nonuniform:
                _n = self._nu_n_steps(num_periods)
            else:
                grid = self._build_grid()
                _n = grid.num_timesteps(num_periods=num_periods)
        _reject_lane_precision("run_distributed_nu" if is_nonuniform else "run_distributed")
        return _DispatchPlan(lane="run_distributed_nu" if is_nonuniform else "run_distributed", n_steps=_n)

    # ---- Non-uniform mesh lane ----
    if is_nonuniform:
        self._reject_upml_on_nonuniform("non-uniform run()")
        _n = n_steps
        if _n is None:
            _n = self._nu_n_steps(num_periods)
        _reject_lane_precision("run_nonuniform")
        return _DispatchPlan(lane="run_nonuniform", n_steps=_n)

    # ---- ADI lane (n_steps resolved by caller from the reused grid) ----
    if self._solver == "adi":
        return _DispatchPlan(lane="run_adi", n_steps=n_steps)

    # ---- Subgridded lane (n_steps resolved by caller — refinement ratio
    # scaling needs the reused grid) ----
    if self._refinement is not None:
        _reject_lane_precision("run_subgridded")
        return _DispatchPlan(lane="run_subgridded", n_steps=n_steps)

    # ---- Uniform lane (n_steps resolved by caller from the reused grid) ----
    return _DispatchPlan(lane="run_uniform", n_steps=n_steps)
