"""rfx.api execute cluster — forward / run dispatch + per-path runners.

Part B Stage 4 (final): the execute-cluster methods of ``Simulation``
extracted into ``_ExecuteMixin``. Pure structural move from
``rfx/api/__init__.py``, no behaviour change. This completes the B.4
five-module package target and fully dissolves the original God class
into a thin facade (``__init__.py``) over five mixin/leaf modules
(``_spec``, ``_compile``, ``_preflight``, ``_sparams``, ``_execute``).

LEAF mixin module — it must NEVER do ``from rfx.api import ...`` or
``from . import ...``; it may import only ``rfx.api._spec`` and external
``rfx.*`` / stdlib / jax / numpy.
"""
from __future__ import annotations

from rfx.stepping import steps_for_periods
from rfx.preflight.line_stub import line_stub_admission as _line_stub_admit
from rfx.api._removed_kwargs import _reject_removed_forward_kwargs
from rfx.api._forward_result import pack_nu_forward_result
from rfx.api._gradient_witness import with_conductivity_record
from rfx.preflight._impl import run_preflight_gate
from rfx._diagnostic_transport import merge_diagnostics

import functools
import math

from rfx import _realized, _grid_metric
from rfx.model import conductors as _conductors, design_box_window
from rfx.model.overrides import apply_material_overrides

import jax
import jax.numpy as jnp
import numpy as np

from rfx.runners._admission import admit_traced_run, refuse_forward_smoothing
from rfx.runners._exchange_interval import validate_exchange_interval
from rfx.sources.sources import GaussianPulse  # noqa: F401  (local import in moved bodies)


from rfx.simulation import SnapshotSpec  # noqa: F401  (run() signature type-hint)
from rfx.ringdown import RingdownSpec  # noqa: F401  (run() signature type-hint)
from rfx.api._spec import (
    ForwardResult,
    Result,
    MATERIAL_LIBRARY,
    _warn_if_nonfinite_result,
)


# ---------------------------------------------------------------------------
# Phase 3 (issue #44 V3 §M6): module-level flag so the distributed=True
# UserWarning fires exactly once per process.  Reset to False on import,
# flipped to True the first time ``Simulation.forward(distributed=True, ...)``
# is invoked.  Lives here (not in __init__.py) because ``forward`` — which
# does ``global _DISTRIBUTED_FIRST_CALL_WARNED`` — was moved into this
# module; the ``global`` statement binds to this module's namespace.
# ---------------------------------------------------------------------------
_DISTRIBUTED_FIRST_CALL_WARNED: bool = False


from rfx.api._refplane_checks import _refplane_conductor_mask, _refplane_reject_msl_ladder_in_plane_zone




from rfx.api._dispatch import _DispatchPlan

from rfx.api._dispatch import (
    _staged_by_an_outer_trace,
    _setup_evaluated_at_trace_time,
    _declaration_setup_at_trace_time,
)
from rfx.api._refplane_checks import _refplane_reject_planes_off_the_uniform_line

class _ExecuteMixin:
    """Execute cluster mixin: forward / run dispatch + per-path runners.

    Mixed into ``Simulation``; all methods stay bound methods on a
    ``Simulation`` instance (resolved via MRO).
    """

    # Runtime-only; set by compute_msl_s_matrix for the duration of a run.
    _dft_plane_regions: dict[str, tuple[int, int, int, int]]

    def _resolve_field_dtype(self):
        """Resolve the requested field dtype before any allocation."""
        from rfx._precision import resolve_field_dtype
        return resolve_field_dtype(self._precision)

    # ---- (2,4) stencil comprehensive API fence (PR-1b) ----

    def _check_stencil_order_supported(self, *, distributed: bool = False) -> None:
        """Reject ``stencil_order=4`` on any unsupported execution lane.

        order=4 is implemented ONLY in the plain uniform-Cartesian core_step
        kernel (``rfx.simulation.update_e``/``update_h``). The nonuniform,
        distributed, distributed-NU, and ADI lanes are SEPARATE runners that
        do not flow through that kernel; CPML/UPML boundaries pull the E/H
        update onto the absorber path. None of those honour ``stencil_order``,
        so a 4th-order request there would silently fall back to 2nd order —
        a CRITICAL correctness failure. This single guard, called at the top
        of every run/forward entry BEFORE any dispatch, guarantees order=4
        can never reach an unsupported runner even if a per-path fence is
        missed. Magnetic images require a second-order kernel.
        """
        from rfx.boundaries.pmc import refuse_waveguide_pmc
        refuse_waveguide_pmc(self)
        magnetic_faces = sorted(self._boundary_spec.pmc_faces())
        if getattr(self, "_stencil_order", 2) != 4:
            return
        if magnetic_faces:
            raise NotImplementedError(
                f"PMC magnetic face(s) {', '.join(magnetic_faces)} with stencil_order=4: "
                "yee._diff_bwd_o far neighbors do not implement the magnetic image; "
                "use stencil_order=2.")
        unsupported = []
        is_nonuniform = (
            self._dz_profile is not None
            or self._dx_profile is not None
            or self._dy_profile is not None
        )
        if is_nonuniform:
            unsupported.append("non-uniform mesh (dx/dy/dz profile)")
        if self._solver == "adi":
            unsupported.append("solver='adi'")
        if self._refinement is not None:
            unsupported.append("subgridding (refinement)")
        if distributed:
            unsupported.append("distributed execution")
        # Absorbing boundary on ANY face (legacy scalar OR per-face spec).
        _absorber = None
        _spec = getattr(self, "_boundary_spec", None)
        if _spec is not None:
            _absorber = _spec.absorber_type  # @property → 'cpml'/'upml'/None
        if _absorber in ("cpml", "upml") or self._boundary in ("cpml", "upml"):
            unsupported.append(
                f"boundary='{_absorber or self._boundary}'"
            )
        if unsupported:
            raise NotImplementedError(
                "stencil_order=4 is only supported on the plain uniform "
                "Cartesian vacuum/dielectric path (uniform mesh, default "
                "'yee' solver, pec/periodic boundary, single device). "
                "Unsupported configuration: " + ", ".join(unsupported)
                + ". Use stencil_order=2 (the default) for these lanes."
            )

    # ---- subgridded run ----

    def _run_subgridded(
        self,
        grid_coarse,
        base_materials_coarse,
        pec_mask_coarse,
        n_steps,
        *,
        diagnostics=(),
        compute_s_params=None,
        s_param_freqs=None,
        s_param_n_steps=None,
        conformal_pec=None,
    ):
        """Run simulation using SBP-SAT subgridding (JIT-compiled)."""
        from rfx.subgridding._notice import require_experimental
        require_experimental(self)
        self._reject_refplane_ports_off_uniform_lane("subgridded (SBP-SAT)",
                                                     compute_s_params)
        from rfx.runners.subgridded import run_subgridded_path
        return run_subgridded_path(
            self,
            grid_coarse,
            base_materials_coarse,
            pec_mask_coarse,
            n_steps,
            compute_s_params=compute_s_params,
            diagnostics=diagnostics,
            s_param_freqs=s_param_freqs,
            s_param_n_steps=s_param_n_steps,
            conformal_pec=conformal_pec,
        )

    # ---- non-uniform mesh run path ----


    def _run_nonuniform(self, *,
                        diagnostics=(),
                        n_steps, compute_s_params=None, preflight=None,
                        conductors=None,
                        s_param_freqs=None,
                        subpixel_smoothing: bool | str = False,
                        checkpoint: bool = False,
                        until_decay: float | None = None,
                        decay_check_interval: int = 50,
                        decay_min_steps: int = 100,
                        decay_max_steps: int = 50_000,
                        decay_energy_consecutive: int = 2,
                        radiated_flux_box: tuple | None = None,
                        flux_env_checks: int = 4,
                        report_every: int | None = None,
                        report_label: str = "",
                        stop_fn=None,
                        stop_interval: int = 250,
                        conformal_pec=None):
        """Run simulation on non-uniform grid with graded dz.

        ``until_decay`` (issue #383) is threaded through to
        ``run_nonuniform_path`` only when the caller has already gated on
        absorbing boundaries. ``run()`` refuses ``until_decay`` for
        closed-boundary NU sims, where the interior energy does not decay. ``decay_monitor_component`` / ``decay_monitor_position``
        are NOT threaded: they parameterize the uniform lane's
        closed/PEC point-field fallback, which has no NU counterpart.
        """
        # Defense-in-depth: the NU runner does not honour stencil_order.
        self._check_stencil_order_supported()
        self._reject_refplane_ports_off_uniform_lane("non-uniform mesh",
                                                     compute_s_params)
        if self._interface_eps == "dual_average" and subpixel_smoothing:
            raise ValueError("interface_eps='dual_average' cannot combine with subpixel_smoothing")
        if subpixel_smoothing == "kottke_pec":
            raise NotImplementedError(
                "subpixel_smoothing='kottke_pec' (Stage 2 unified PEC) "
                "is not supported on the non-uniform mesh path. "
                "Drop the dx/dy/dz profile or use the legacy "
                "``Boundary(conformal=True)`` Stage 1 path on a "
                "uniform mesh. Tracked at "
                "https://github.com/bk-squared/rfx/issues."
            )
        from rfx.runners.nonuniform import run_nonuniform_path
        return run_nonuniform_path(
            self,
            preflight=preflight,
            conductors=conductors,
            diagnostics=diagnostics,
            n_steps=n_steps,
            compute_s_params=compute_s_params,
            s_param_freqs=s_param_freqs,
            subpixel_smoothing=subpixel_smoothing,
            checkpoint=checkpoint,
            until_decay=until_decay,
            report_every=report_every,
            report_label=report_label,
            decay_check_interval=decay_check_interval,
            decay_min_steps=decay_min_steps,
            decay_max_steps=decay_max_steps,
            decay_energy_consecutive=decay_energy_consecutive,
            radiated_flux_box=radiated_flux_box,
            flux_env_checks=flux_env_checks,
            **({} if stop_fn is None else {"stop_fn": stop_fn, "stop_interval": stop_interval}),
            conformal_pec=conformal_pec,
        )

    @staticmethod
    def _warn_run_sparams_if_nonpassive(result) -> None:
        """Apply the shared lumped/wire advisory to an assembled run result."""
        s_params = getattr(result, "s_params", None)
        freqs = getattr(result, "freqs", None)
        if s_params is None or freqs is None:
            return
        from rfx.probes.probes import warn_if_nonpassive_lumped_s11
        from rfx.measurement import observed_port_s
        pairs = getattr(result, "wire_port_sparams", None)
        warn_if_nonpassive_lumped_s11(
            observed_port_s(s_params, pairs) if pairs else s_params, freqs,
            extractor="run(compute_s_params=True)")

    def _warn_postrun_energy_witness(
        self,
        result,
        *,
        fixed_num_periods: bool,
        until_decay: float | None,
    ) -> None:
        from rfx.measurement import warn_postrun_energy_witness
        return warn_postrun_energy_witness(self, result, fixed_num_periods=fixed_num_periods, until_decay=until_decay)

    def _run_settling_witness(self, result):
        from rfx.api._settling import run_settling_witness
        return run_settling_witness(self, result)

    def _settling_source_end(self, result, records=None):
        from rfx.api._settling import settling_source_end
        return settling_source_end(self, result, records)

    def _settling_probe_selection(self, result):
        """Numeric probe selection metadata, with the #1090 drive flag.

        The grid is read off the RESULT (``Result.grid`` /
        ``ForwardResult.grid``) -- the grid the run actually built, not a
        re-derivation -- so the co-location test uses the same node spine
        that placed the sources and the probes. A lane that returns no grid
        simply carries no flags: the witness then behaves exactly as it did
        before this check existed rather than guessing at cell identity.
        """
        from rfx.probes.settling import (
            drive_cells, probe_record_info, source_dominated_columns,
        )

        grid = getattr(result, "grid", None)
        dominated = source_dominated_columns(
            grid, self._probes,
            drive_cells(grid, getattr(self, "_ports", ()),
                        getattr(self, "_msl_ports", ())),
        )
        return probe_record_info(
            getattr(result, "time_series", None), self._probes,
            getattr(self, "_internal_probe_indices", ()),
            source_dominated=dominated,
        )

    def _attach_run_settling_witness(self, result, *, n_steps=None,
                                     num_periods=None, context="run"):
        from rfx.measurement import attach_run_settling_witness
        return attach_run_settling_witness(self, result, n_steps=n_steps, num_periods=num_periods, context=context)

    def _warn_decay_param_sanity(self, *, until_decay, decay_min_steps,
                                 decay_max_steps) -> None:
        """Behaviour-NEUTRAL sanity advisories on the decay-stop params.

        Warn-only, single ``run()``-level site so both lanes that honour
        ``until_decay`` (uniform and absorbing-boundary non-uniform) are
        covered by one check (post-#392 review):

        * ``decay_min_steps > decay_max_steps`` — every check index lies
          past the table end, so zero energy checks occur and the run is
          a forced ``decay_max_steps`` run (the stop can never fire).
        * ``until_decay >= 1`` — ``W < until_decay * reference`` holds at
          every check where W is not still rising, so the stop can fire
          right after the drive end and ``decay_min_steps``.
          (``until_decay == 0.0`` is the documented forced-N escape on
          the NU lane and is deliberately not warned about.)
        """
        import warnings

        if until_decay is None:
            return
        if decay_min_steps > decay_max_steps:
            warnings.warn(
                f"decay_min_steps ({decay_min_steps}) > decay_max_steps "
                f"({decay_max_steps}): zero energy checks will occur — "
                "the decay stop can never fire and this is a forced "
                "decay_max_steps run. Lower decay_min_steps or raise "
                "decay_max_steps.",
                UserWarning, stacklevel=3,
            )
        try:
            _ud = float(until_decay)
        except (TypeError, ValueError):
            return
        if _ud >= 1.0:
            warnings.warn(
                f"until_decay={until_decay} >= 1: the energy criterion "
                "(W < until_decay * reference) holds at every check where W "
                "is not still rising, so the stop can fire right after the "
                "drive end and decay_min_steps; use a fraction such as 1e-3.",
                UserWarning, stacklevel=3,
            )

    def _reject_refplane_ports_off_uniform_lane(self, path_name: str,
                                                compute_s_params) -> None:
        """Fail loudly when reference-plane ports hit a non-uniform lane.

        The issue #313 plane V/I accumulators are wired only into the
        uniform single-device production scan; silently returning legacy
        port-cell off-diagonals on another lane would defeat the opt-in.
        """
        if compute_s_params is False:
            return
        if any(getattr(pe, "reference_plane_cells", None) is not None
               for pe in self._ports):
            raise NotImplementedError(
                "add_port(reference_plane_cells=...) S-parameters are only "
                f"supported on the uniform single-device run() lane, not "
                f"the {path_name} lane. Drop the opt-in or use a uniform "
                "mesh (issue #313)."
            )

    @staticmethod
    def _refuse_unsupported_run_kwargs(path_name: str,
                                       unsupported_kwargs: dict,
                                       *,
                                       instead: str | dict | None,
                                       reason_overrides: dict | None = None,
                                       remedy_overrides: dict | None = None,
                                       entry: str = "Simulation.run()") -> None:
        """Refuse a ``Simulation.run`` argument that a lane does not implement.

        ``unsupported_kwargs`` maps each ``run()`` argument the
        ``path_name`` lane does not implement to the value the caller
        passed. A value that does not ask for the feature (the entries of
        ``silent_values``, e.g. ``conformal_pec=False``) passes. Any other
        value raises ``NotImplementedError`` naming the argument, the
        lane, why the lane cannot do it and what to do instead. Running
        without it would not be the run that was asked for: most of these
        change the result (a staircase PEC instead of the conformal
        boundary, a fixed ``n_steps`` instead of a run to decay, no
        S-matrix), and ``checkpoint`` changes the memory of a gradient,
        not its value. Until 2.0 these lanes ran on behind a
        ``UserWarning``.

        ``report_every`` is the one argument that only warns: it prints
        progress and changes no result, so a lane without it says so and
        runs on.

        ``instead`` names how to reach a lane that implements the
        arguments (e.g. ``"run on a uniform mesh"``), or is ``None`` when
        there is no such lane. A dict gives it per argument, with ``"*"``
        for the rest, where one lane change does not reach every argument
        (a distributed non-uniform run that drops ``devices=`` still lands
        on the non-uniform lane). ``reason_overrides`` replaces the shared
        per-argument reason with a lane-accurate one (the closed-boundary
        non-uniform lane refuses ``until_decay`` for a different reason
        than the distributed lane does); ``remedy_overrides`` does the same
        for what to do instead. ``entry`` names the call the user made, for
        the S-parameter calculators that share this rule.
        """
        import warnings as _w
        # Per-kwarg values that do not ask for the feature. ``None`` is
        # compared by identity, so an array-valued argument never reaches
        # an elementwise ``==``.
        silent_values = {
            "subpixel_smoothing": (False,),
            "checkpoint": (False,),
            "snapshot": (None,),
            "until_decay": (None,),
            "conformal_pec": (False,),
            "compute_s_params": (None, False),
            "s_param_freqs": (None,),
            "s_param_n_steps": (None,),
        }
        reasons = {
            "subpixel_smoothing":
                "the smoothed per-component permittivity tensor is not "
                "wired into this lane's E update, so interfaces would get "
                "scalar eps",
            "checkpoint":
                "this lane's time loop has no checkpointed scan, so "
                "reverse-mode AD would store the full tape",
            "snapshot":
                "this lane records no field snapshots",
            "until_decay":
                "this lane runs exactly n_steps; the decay stop exists on "
                "the uniform lane and on the non-uniform lane with "
                "cpml/upml boundaries",
            "conformal_pec":
                "the Dey-Mittra conformal weights are wired only into the "
                "uniform Yee lane, so PEC would be staircased",
            "compute_s_params":
                "S-matrix assembly is not plumbed through this lane",
            "s_param_freqs":
                "S-matrix assembly is not plumbed through this lane",
            "s_param_n_steps":
                "S-matrix assembly is not plumbed through this lane",
            "report_every":
                "host-side progress chunking (issue #667) is wired on the "
                "uniform and non-uniform lanes only",
        }
        remedies = {
            "subpixel_smoothing": "drop subpixel_smoothing",
            "checkpoint": "pass checkpoint=False",
            "snapshot": "drop snapshot= and read Result.state or add_probe() "
                        "time series",
            "until_decay": "set n_steps or num_periods",
            "conformal_pec": "pass conformal_pec=False and drop "
                             "Boundary(conformal=True) (staircase PEC)",
            "compute_s_params": "drop the S-parameter request",
            "s_param_freqs": "drop the S-parameter request",
            "s_param_n_steps": "drop s_param_n_steps and set n_steps to "
                               "the record length you want",
        }
        if reason_overrides:
            reasons.update(reason_overrides)
        if remedy_overrides:
            remedies.update(remedy_overrides)

        def _asks(kw, val) -> bool:
            silent = silent_values.get(kw, (None,))
            if val is None:
                return None not in silent
            if isinstance(val, (bool, int, float, str, np.generic)):
                return val not in silent
            return True

        def _shown(val) -> str:
            text = repr(val)
            return text if len(text) <= 40 else f"<{type(val).__name__}>"

        refused = []
        for kw, val in unsupported_kwargs.items():
            if kw == "report_every" or not _asks(kw, val):
                continue
            other = (instead.get(kw, instead.get("*"))
                     if isinstance(instead, dict) else instead)
            refused.append(
                f"  {kw}={_shown(val)}: {reasons.get(kw, 'not implemented')}. "
                f"Instead: {remedies.get(kw, 'drop it')}"
                + (f", or {other}." if other else ".")
            )
        if refused:
            raise NotImplementedError(
                f"{entry} refuses {len(refused)} argument(s) the "
                f"{path_name} lane does not implement; running on without "
                "them would not be the run that was asked for:\n"
                + "\n".join(refused)
            )
        val = unsupported_kwargs.get("report_every")
        if val is not None:
            _w.warn(
                f"report_every={val!r} is ignored on the {path_name} run "
                f"path ({reasons['report_every']}); the run goes on without "
                "progress lines and its result is unchanged.",
                UserWarning, stacklevel=3,
            )

    from rfx.runners._admission import _refuse_conformal_boundary

    @staticmethod
    def _s_param_n_steps_off_record(s_param_n_steps, n_steps, until_decay):
        """``s_param_n_steps`` where it differs from the main run's record.

        For a lane that reads its S-parameters from the main run's own port
        record, that record is ``n_steps`` long, or ends at the decay stop.
        An equal value asks for nothing and comes back as ``None``.
        """
        if s_param_n_steps is None:
            return None
        if (until_decay is None and n_steps is not None
                and int(s_param_n_steps) == int(n_steps)):
            return None
        return s_param_n_steps

    def _port_sparameter_entries(self) -> list:
        """Return ``add_port`` entries that are actual impedance ports.

        ``add_source`` reuses ``_PortEntry`` with ``impedance=0`` as a
        source-only sentinel, so S-parameter request validation must not
        count every ``self._ports`` entry as a calibrated port.
        """

        return [pe for pe in self._ports if pe.impedance > 0.0]

    def _auto_configure_mesh(self):
        """Compute the host-side auto-mesh proposal; _resolve_mesh owns it."""
        from rfx.auto_config import auto_configure

        geometry_pairs = [
            (entry.shape, entry.material_name)
            for entry in self._geometry
        ]
        materials_dict = {}
        for name, spec in self._materials.items():
            materials_dict[name] = {
                "eps_r": spec.eps_r,
                "sigma": spec.sigma,
            }
        # Include library materials used by geometry but not explicitly registered
        for entry in self._geometry:
            mname = entry.material_name
            if mname not in materials_dict and mname in MATERIAL_LIBRARY:
                materials_dict[mname] = MATERIAL_LIBRARY[mname]

        # Thin conductors are PEC/lossy sheets absent from self._geometry
        # (issue #371 Bug 2). They add no wavelength constraint, but their
        # IN-PLANE feature size must feed the feature-based dx. Register each
        # under a synthetic material so analyze_features() folds its bbox dims
        # into min_thickness while classifying it by sigma (PEC vs lossy), NOT
        # as a dielectric: sigma_bulk>=1e6 => has_pec (kept out of max_eps_r),
        # and a sheet's zero z-extent (< 1e-12, plus the sigma<pec_threshold
        # guard) keeps it out of z_features (no false non-uniform-z). The
        # zero z-extent is dropped by the nonzero-dims filter, so min_thickness
        # resolves to the in-plane extent.
        for i, tc in enumerate(self._thin_conductors):
            tc_mat = f"__thin_conductor_{i}__"
            materials_dict[tc_mat] = {"eps_r": tc.eps_r, "sigma": tc.sigma_bulk}
            geometry_pairs.append((tc.shape, tc_mat))

        config = auto_configure(
            geometry=geometry_pairs,
            freq_range=(self._freq_max / 10, self._freq_max),
            materials=materials_dict,
            boundary=self._boundary,
            # #931 §1.9: a sheet's PLANE is a z feature even though its zero
            # thickness is not — the profile must leave a mesh line there.
            thin_conductors=self._thin_conductors,
        )

        return config

    from rfx.runners._admission import _reject_upml_on_nonuniform

    def _auto_preflight(
        self, *, skip: bool = False, context: str = "forward",
        check_ntff: bool | str = True, conductors=None,
        prepare=False, distributed=False, tfsf_material_overrides=(),
    ) -> _conductors.PreparedSolve | _conductors.PreflightReport:
        """Delegate validation and optional solve ownership to the conductors layer."""
        return _conductors.auto_preflight(self, skip=skip, context=context, check_ntff=check_ntff, conductors=conductors, prepare=prepare, distributed=distributed, tfsf_material_overrides=tfsf_material_overrides)

    def _run_preflight_gate(self, issues, *, context: str, stacklevel: int = 3) -> None:
        return run_preflight_gate(issues, context, stacklevel)

    from rfx.runners.adi import _run_adi_from_materials

    from rfx.runners.uniform_forward import _forward_from_materials

    _pack_nu_forward_result = staticmethod(pack_nu_forward_result)

    @_declaration_setup_at_trace_time
    def _forward_nonuniform_from_materials(
        self,
        *,
        subpixel_smoothing=False,
        diagnostics=(),
        preflight=None,
        conductors=None,
        eps_override: jnp.ndarray | None = None,
        sigma_override: jnp.ndarray | None = None,
        pec_mask_override: jnp.ndarray | None = None,
        pec_occupancy_override: jnp.ndarray | None = None,
        pinned_sheet_layout: tuple | None = None,
        design_box: object | None = None,
        n_steps: int,
        port_s11_freqs: object | None = None,
        checkpoint: bool = True,
        emit_time_series: bool = True,
        checkpoint_every: int | None = None,
        n_warmup: int = 0,
    ) -> ForwardResult:
        """Differentiable forward on the non-uniform mesh path.

        Routes through ``run_nonuniform_path`` with optimisation overrides
        applied after material assembly, then repackages the returned
        ``Result`` into the minimal ``ForwardResult`` schema via the shared
        :meth:`_pack_nu_forward_result` lane helper.

        When ``checkpoint`` is True (the default), the NU scan body is
        wrapped in ``jax.checkpoint`` so reverse-mode AD memory scales
        with ``sqrt(n_steps)`` instead of ``n_steps``.
        """
        # Defense-in-depth: the NU runner does not honour stencil_order.
        self._check_stencil_order_supported()
        from rfx.runners.nonuniform import run_nonuniform_path

        # A grid-shaped override is (cells + 1) per axis, because N cells are
        # bounded by N+1 E-nodes (#562) — the same shape the uniform path has
        # always wanted. Caller code that sized arrays from the PROFILE length
        # is one sample short, and without this check the mismatch surfaces
        # deep inside XLA as an opaque broadcast TypeError (#562 review F9).
        _overrides = (("eps_override", eps_override),
                      ("sigma_override", sigma_override),
                      ("pec_mask_override", pec_mask_override),
                      ("pec_occupancy_override", pec_occupancy_override))
        # Build the grid ONLY when there is an array to check, so the common
        # override-free forward() does not pay for a second grid build
        # (#568 observation 1 — negligible against the solve, but free to skip).
        if any(_a is not None and hasattr(_a, "shape") for _n, _a in _overrides):
            _grid = self._build_nonuniform_grid()
            for _name, _arr in _overrides:
                if _arr is None or not hasattr(_arr, "shape"):
                    continue
                if tuple(_arr.shape) == tuple(_grid.shape):
                    continue
                raise ValueError(
                    f"{_name} has shape {tuple(_arr.shape)} but this "
                    f"non-uniform grid is {tuple(_grid.shape)}. Grid-shaped "
                    f"arrays are one sample larger per axis than the cell "
                    f"counts / profile lengths you requested: N cells are "
                    f"bounded by N+1 field nodes (#562). Build the array from "
                    f"the grid shape rather than from the profile length."
                )

        result = run_nonuniform_path(
            self,
            subpixel_smoothing=subpixel_smoothing,
            preflight=preflight,
            conductors=conductors,
            diagnostics=diagnostics,
            n_steps=n_steps,
            s_param_freqs=(
                None if port_s11_freqs is None else jnp.asarray(port_s11_freqs, dtype=jnp.float32)
            ),
            eps_override=eps_override,
            sigma_override=sigma_override,
            pec_mask_override=pec_mask_override,
            pec_occupancy_override=pec_occupancy_override,
            pinned_sheet_layout=pinned_sheet_layout,
            design_box=design_box,
            checkpoint=checkpoint,
            emit_time_series=emit_time_series,
            checkpoint_every=checkpoint_every,
            n_warmup=n_warmup,
            lane="fwd_nonuniform",
        )
        return self._pack_nu_forward_result(
            diagnostics=result.diagnostics,
            time_series=result.time_series,
            ntff_data=result.ntff_data,
            ntff_box=result.ntff_box,
            grid=result.grid,
            s_params=getattr(result, "s_params", None),
            freqs=getattr(result, "freqs", None),
            dft_planes=getattr(result, "dft_planes", None),
            flux_monitors=result.flux_monitors,
            waveguide_ports=result.waveguide_ports,
            waveguide_sparams=result.waveguide_sparams,
            wire_port_sparams=getattr(result, "wire_port_sparams", None),
            sparam_time_records=getattr(result, "sparam_time_records", None),
            sparam_read_ids=getattr(result, "sparam_read_ids", None),
            dft_time_records=getattr(result, "dft_time_records", None),
            dt=getattr(result, "dt", None),
            current_moment_data=getattr(result, "current_moment_data", None),
            current_moment_monitor=getattr(result, "current_moment_monitor", None),
        )

    def distributed_override_layout(self, devices=None):
        """Return ``(shape, NamedSharding)`` for distributed forward designs.

        The global shape is ``(nx_padded, ny, nz)`` with no ghosts, where
        ``nx_padded`` rounds the NU grid's x extent up to a multiple of the
        device count. Sharding is ``P("x")`` on the ordered ``devices``
        (default: all global JAX devices). Every process must use the same
        list. Construct large designs with ``jax.make_array_from_callback``
        or ``jax.make_array_from_single_device_arrays`` using this layout;
        no whole-domain device allocation is necessary. Forward ignores
        alignment-pad values, and their returned gradients are zero.
        """
        from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
        from rfx.runners.distributed_nu import build_sharded_nu_grid

        devices = list(jax.devices() if devices is None else devices)
        if not devices or len(set(devices)) != len(devices) or any(
                device not in jax.devices() for device in devices):
            raise ValueError("devices must be a nonempty ordered list of distinct JAX devices")
        grid = build_sharded_nu_grid(self._build_nonuniform_grid(), len(devices))
        return ((grid.nx_padded, grid.ny, grid.nz),
                NamedSharding(Mesh(np.array(devices), ("x",)), P("x")))

    def shard_distributed_override(self, array, devices=None, *, pad_value=0.0):
        """Place a concrete local whole-domain design in the forward layout.

        This convenience initializer copies addressable x slabs from a host
        array; it is not an autodiff operation. Differentiate forward with
        respect to the returned global array. For designs too large for one
        host, construct shards directly using ``distributed_override_layout``.
        On multiple processes each caller supplies the same concrete array.
        """
        _, sharding = self.distributed_override_layout(devices)
        host = np.asarray(array)
        expected = self._build_nonuniform_grid().shape
        if host.shape != expected:
            raise ValueError(f"local override must have grid shape {expected}; got {host.shape}")

        from rfx.stepping.slab import Slab, Fill, cut
        return cut(host, Slab(expected[0], sharding.mesh.size, 0),
                   Fill(pad_value, pad_value), mesh=sharding.mesh)

    def _execute_distributed_nonuniform_from_materials(self, **kwargs):
        from rfx.runners._distributed_nu_execute import execute_from_materials
        return execute_from_materials(self, **kwargs)

    def _forward_distributed_nonuniform_from_materials(self, **kwargs):
        from rfx.runners._distributed_nu_execute import forward_from_materials
        return forward_from_materials(self, **kwargs)

    # ---- unified lane dispatch (W6.3) ----

    def _nu_n_steps(self, num_periods: float) -> int:
        """Resolve the timestep count for a non-uniform / distributed-NU
        lane from a throwaway NU grid build.

        Pure: ``_build_nonuniform_grid()`` synthesises any missing dz/dx/dy
        profile internally without mutating sim state (W1.3). The single
        rule is :func:`rfx.stepping.step_count.steps_for_periods`.
        """
        grid = self._build_nonuniform_grid()
        return steps_for_periods(grid.dt, self._freq_max, num_periods)

    from rfx.api._dispatch import _dispatch_plan

    # ---- forward (differentiable) ----

    def _resolve_design_box_override(
        self,
        grid,
        *,
        design_box,
        design_eps_override,
        design_sigma_override,
        design_occupancy_override,
        eps_override,
        sigma_override,
        mu_r_override,
        pec_occupancy_override,
        debye_spec,
        lorentz_spec,
        kerr_chi3,
        holds_ports=False,
    ):
        """Turn the ``forward(design_box=...)`` arguments into specs (#1179/#1183).

        Resolves the declared corners against the grid the solve builds and
        rejects every combination this lane does not carry. The checks that
        need the RESOLVED step context — the absorber depth, the port and
        source cells, the anisotropy flags — belong to
        ``rfx.model.resolve_design_box`` /
        ``_resolve_design_occupancy`` and run there; these are the ones that
        can be decided from the call itself.

        Returns ``(box spec or None, occupancy spec or None)``: one design
        box, and the design variable it carries — a permittivity (#1179) or
        a PEC occupancy (#1183).

        ``holds_ports`` (``forward(design_box_holds_ports=True)``) lets the
        permittivity box contain lumped and wire ports: every edge a port
        drives or loads that lies in the box's update window is listed in
        ``DesignBoxSpec.held_edges`` and keeps the coefficients the drawn
        materials give it.
        """
        from rfx.simulation import (
            DesignBoxSpec, DesignOccupancySpec,
        )

        _want_eps = design_eps_override is not None
        _want_occ = design_occupancy_override is not None
        if holds_ports and not _want_eps:
            raise ValueError(
                "forward(design_box_holds_ports=True) lets a PERMITTIVITY "
                "design box (design_box with design_eps_override) contain "
                "lumped and wire ports. It has no meaning without one, and "
                "an occupancy box (design_occupancy_override) does not carry "
                "it: the port setup clears the occupancy around its cell, "
                "which the box would overwrite.")
        if design_box is None or not (_want_eps or _want_occ):
            raise ValueError(
                "forward(design_box=...) and one of "
                "forward(design_eps_override=...) / "
                "forward(design_occupancy_override=...) are used together "
                "(#1179/#1183): the box says WHICH cells and the array says "
                "what the design variable is there. Got design_box="
                f"{'set' if design_box is not None else 'None'}, "
                f"design_eps_override={'set' if _want_eps else 'None'}, "
                f"design_occupancy_override={'set' if _want_occ else 'None'}.")
        if _want_eps and _want_occ:
            raise NotImplementedError(
                "forward(design_eps_override=...) and "
                "forward(design_occupancy_override=...) in one call are not "
                "supported (#1183). Each is exact on its own against its "
                "whole-grid counterpart; the two together have no equality "
                "gate, and the permittivity box is fenced against an "
                "occupancy anyway. Differentiate one at a time, or use "
                "eps_override / pec_occupancy_override.")
        if design_sigma_override is not None and not _want_eps:
            raise ValueError(
                "forward(design_sigma_override=...) is the conductivity of a "
                "PERMITTIVITY design box (#1179) and needs "
                "design_eps_override with it.")

        bounds = self._design_box_bounds_from_corners(grid, design_box)

        from rfx.model import refuse_design_films
        refuse_design_films(self, grid, bounds)

        if not _want_eps:
            # The occupancy box reads no material array and runs after the
            # E update and the absorber, so the material-path fences below
            # do not apply to it: whatever computed E, both formulations
            # arrive at this slot with the same field.
            return None, DesignOccupancySpec(
                bounds=bounds, occupancy=design_occupancy_override)

        _collides = [
            name for name, value in (
                ("eps_override", eps_override),
                ("sigma_override", sigma_override),
                ("mu_r_override", mu_r_override),
            ) if value is not None
        ]
        if _collides:
            raise ValueError(
                "forward(design_box=...) does not combine with "
                + " / ".join(_collides) + " (#1179). The design box exists to "
                "keep the whole-grid material arrays CONSTANT; a traced "
                "override puts them back on the AD tape, which is the cost "
                "the box removes. Put the design values in "
                "design_eps_override, or drop the box and use the whole-grid "
                "override alone.")

        if debye_spec is not None or lorentz_spec is not None:
            raise NotImplementedError(
                "forward(design_box=...) does not support Debye/Lorentz "
                "dispersive materials (#1179): the dispersive E update runs "
                "its own ADE coefficients, which the box redo would "
                "overwrite. Use eps_override.")
        if kerr_chi3 is not None:
            raise NotImplementedError(
                "forward(design_box=...) does not support Kerr chi3 "
                "materials (#1179): the reactive Kerr correction rescales "
                "the E increment using the background eps_r. Use "
                "eps_override.")
        if pec_occupancy_override is not None:
            raise NotImplementedError(
                "forward(design_box=...) does not combine with "
                "pec_occupancy_override (#1179): the relaxed-conductor lane "
                "can route the E update through a Kottke inverse-eps tensor "
                "built from the background permittivity. Use eps_override "
                "for a design region that is itself the conductor.")


        # A port whose cells are in the box. The step-level check
        # (``rfx.model.resolve_design_box``) sees only the ports that
        # left a source or an S-param accumulator behind, so a PASSIVE load
        # port — which folds its impedance into sigma at its own cells and
        # leaves neither — would pass it. Checked here from the
        # declarations, where every port is visible whatever it does later.
        #
        # A port's cells are its EDGES, not its end NODES. An extent from
        # node n0 to node n1 drives the edges n0 .. n1 - 1, so node n1 is
        # only the far terminal and the plane it sits on carries no port
        # cell. Comparing against n1 refused a design box on the port's end
        # plane — exactly where a patch fed by a via lives. The edge span is
        # read from ``rfx.sources.sources.wire_port_edge_span`` (the ONE
        # spelling both runners rasterize with) rather than re-derived here:
        # a sub-cell extent whose two ends snap to one node drives the edge
        # on the side the extent occupies, which ``hi - 1`` cannot know. The
        # step-level fence (``rfx.model.resolve_design_box``) reads the
        # port's ``live_cells`` on the uniform lane and, on the graded lane,
        # the mid cell plus the excited edges through its ``sources`` entry.
        #
        # With ``holds_ports`` a port may be inside: every one of its edges in
        # the box's update window (the box plus one cell on each plus side,
        # where the design material also reaches) is HELD -- it keeps the
        # coefficient the drawn materials give it, the one its load and drive
        # were built with. The step-level resolver checks every port, source
        # and lumped stamp it can see against this list.
        from rfx.sources.sources import wire_port_edge_span
        _axis_of = {"ex": 0, "ey": 1, "ez": 2}
        _w = design_box_window(bounds, tuple(grid.shape))[0]
        _held = set()

        def _in_window(cell):
            return all(_w[2 * d] <= cell[d] < _w[2 * d + 1] for d in range(3))

        for _pe in self._ports:
            if _pe.impedance == 0.0:
                # A plain soft source: caught as a source cell at the step --
                # except on a held port edge, where a port's drive is
                # accepted, so it is refused here whatever it sits on.
                _src = _grid_metric.field_index(grid, _pe.position, _pe.component)
                if holds_ports and _in_window(_src):
                    raise ValueError(
                        f"the design box (cells {bounds}) holds a soft "
                        f"source (component {_pe.component!r} at "
                        f"{_pe.position}). design_box_holds_ports holds a "
                        f"lumped or wire port's edges; a soft source cannot "
                        f"be held. Move the box off the source, or use "
                        f"eps_override.")
                continue
            _axis = _axis_of[_pe.component]
            _lo = list(self._design_box_index_of(grid, _pe.position) if _pe.extent is not None else _grid_metric.field_index(grid, _pe.position, _pe.component))
            _cell_lo = list(_lo)
            _cell_hi = list(_lo)
            if getattr(_pe, "extent", None) is not None:
                _end = list(_pe.position)
                _end[_axis] += _pe.extent
                from rfx._periodic import interval_indices
                _start_idx, _end_idx = interval_indices(
                    grid, tuple(float(v) for v in _pe.position),
                    tuple(float(v) for v in _end))
                _n0, _n1 = sorted((_start_idx[_axis], _end_idx[_axis]))
                _cell_lo[_axis], _cell_hi[_axis] = wire_port_edge_span(
                    grid, _axis, _n0, _n1, _pe.position[_axis], _end[_axis])
            if holds_ports:
                for _a in range(_cell_lo[_axis], _cell_hi[_axis] + 1):
                    _edge = list(_lo)
                    _edge[_axis] = _a
                    if _in_window(_edge):
                        _held.add((_axis, *(int(v) for v in _edge)))
                continue
            if all(bounds[2 * d] <= _cell_hi[d]
                   and _cell_lo[d] < bounds[2 * d + 1]
                   for d in range(3)):
                raise ValueError(
                    f"the design box (cells {bounds}) holds port cells "
                    f"(component {_pe.component!r} at {_pe.position}). A "
                    f"port's impedance fold and its drive coefficient are "
                    f"built from the background permittivity before the time "
                    f"loop starts, so the design permittivity would not reach "
                    f"them (#1179). Move the box off the port, pass "
                    f"design_box_holds_ports=True to keep the port inside on "
                    f"its own drawn coefficients (its edges then take no "
                    f"design value), or use eps_override.")

        return DesignBoxSpec(
            bounds=bounds,
            eps_r=design_eps_override,
            sigma=design_sigma_override,
            held_edges=tuple(sorted(_held)),
        ), None

    @staticmethod
    def _design_box_index_of(grid, position):
        """Cell index of a position, on the uniform or the graded grid.

        ``Grid`` carries the method; ``NonUniformGrid`` is a pytree-registered
        NamedTuple and its resolver is a module function that walks the
        cumulative cell edges (#1183).
        """
        if hasattr(grid, "position_to_index"):
            return grid.position_to_index(tuple(float(v) for v in position))
        from rfx.nonuniform import position_to_index as _nu_pos_to_idx
        return _nu_pos_to_idx(grid, tuple(float(v) for v in position))

    def _design_box_bounds_from_corners(self, grid, design_box):
        """Half-open cell bounds of a declared design box (#1179).

        Same index resolution topology_optimize uses (rfx/topology.py):
        nearest index of each corner, both ends inclusive, so a corner pair
        inside the interior selects the same cells through either entry
        point. The two differ on a corner OUTSIDE it: topology_optimize
        CLAMPS to the interior and runs the clamped region, this path
        raises (the absorber fence in the step-level resolver). The REALIZED
        cell counts are what the arrays must match -- the declared extent is
        not it at any dx.
        """
        corner_lo = getattr(design_box, "corner_lo", None)
        corner_hi = getattr(design_box, "corner_hi", None)
        if corner_lo is None or corner_hi is None:
            try:
                corner_lo, corner_hi = design_box
            except (TypeError, ValueError):
                raise ValueError(
                    "forward(design_box=...) takes a (corner_lo, corner_hi) "
                    "pair of (x, y, z) positions in metres, or an object "
                    "carrying corner_lo/corner_hi (DesignRegion, "
                    f"TopologyDesignRegion). Got {design_box!r}.") from None
        from rfx._periodic import interval_indices
        lo_idx, hi_idx = interval_indices(
            grid, tuple(float(v) for v in corner_lo), tuple(float(v) for v in corner_hi))
        return tuple(
            v for axis in range(3)
            for v in (int(lo_idx[axis]), int(hi_idx[axis]) + int(
                not ('xyz'[axis] in getattr(grid, 'periodic_axes', '')
                     and hi_idx[axis] == grid.shape[axis])))
        )

    @_declaration_setup_at_trace_time
    def forward(
        self,
        *,
        eps_override: jnp.ndarray | None = None,
        sigma_override: jnp.ndarray | None = None,
        mu_r_override: jnp.ndarray | None = None,
        design_box: tuple | object | None = None,
        design_eps_override: jnp.ndarray | None = None,
        design_sigma_override: jnp.ndarray | None = None,
        design_occupancy_override: jnp.ndarray | None = None,
        design_box_holds_ports: bool = False,
        pec_mask_override: jnp.ndarray | None = None,
        pec_occupancy_override: jnp.ndarray | None = None,
        subpixel_smoothing: bool | str = False,
        pinned_sheet_layout: tuple | None = None,
        n_steps: int | None = None,
        num_periods: float = 20.0,
        checkpoint: bool | None = None,
        gradient: str = "autodiff",
        checkpoint_segments: int | None = None,
        emit_time_series: bool = True,
        checkpoint_every: int | None = None,
        n_warmup: int = 0,
        skip_preflight: bool = False,
        distributed: bool = False,
        devices: list | None = None,
        exchange_interval: int = 1,
        port_s11_freqs: object | None = None,
        rlc_values_override: dict | None = None,
        ringdown: RingdownSpec | None = None,
        **_removed_kwargs,
    ) -> ForwardResult:
        """Run a minimal differentiable forward simulation.

        This path is designed for ``jax.grad`` / ``jax.value_and_grad`` and
        intentionally returns only the observables needed by differentiable
        objectives instead of the broader stateful :meth:`run` result.

        Parameters
        ----------
        subpixel_smoothing : bool or str
            False: staircase; True: diagonal, first order on curved interfaces (worst at eps ~30+).
            "full_tensor": full anisotropic averaging, uniform single-device; synthetic eps-4 sphere extinction measured second order over three meshes.
            Farjadpour 2006; Oskooi 2009; Werner-Bauer-Cary 2013 node form. "kottke_pec": unified PEC inverse tensor.
        gradient : {"autodiff", "adjoint"}
            Opt-in settled-spectrum reciprocity design eps adjoint.
            Design sigma overrides are refused (#1424): on lossless design cells the conductivity gradient has no settled value in a finite record
            (a static sensitivity remains although the field decays); this adjoint returns settled-spectrum gradients only. Fixed lossy materials are allowed.
            Stores design-edge DFTs and runs a second ordinary forward.
            ``ForwardResult.adjoint_settling`` reads the primal design-edge max|E| over the last period of the lowest bin, capped at the record length, divided by the record maximum.
            Above about 1e-2 means unsettled; small proves nothing: it follows the strongest field and says nothing about the adjoint record.
            Use ``gradient_record_length_witness`` to witness an adjoint gradient.
            Uniform Yee design_box only; unsupported paths raise. Checkpoint
            options are unused by the adjoint (its storage does not depend on them).
        eps_override : jnp.ndarray or None
            Replacement permittivity array with shape ``grid.shape``.
            With ``distributed=True``, eps/sigma/occupancy overrides also accept
            x-sharded global arrays from :meth:`distributed_override_layout`.
            That form is required across processes; gradients keep its sharding.
            Use :meth:`shard_distributed_override` for a concrete local initializer.
        sigma_override : jnp.ndarray or None
            Replacement conductivity array with shape ``grid.shape``.
        mu_r_override : jnp.ndarray or None
            Replacement relative permeability array with shape ``grid.shape``.
            Differentiable (the H-update coefficient is ``dt/(mu_r*MU_0*dx)``), so
            ``jax.grad`` flows w.r.t. a magnetic-material design variable (ferrite /
            magnetic-RAM inverse design). Uniform single-device lane only; a
            non-uniform mesh or ``distributed=True`` raises (it would otherwise be
            silently dropped — a zero-gradient no-op).
        design_box : (corner_lo, corner_hi) or a DesignRegion, or None
            Issue #1179. Bounding box, in metres, of a design region whose
            permittivity is handed in separately as *design_eps_override*
            instead of being written into a whole-grid ``eps_override``.
            Accepts a ``(corner_lo, corner_hi)`` pair or any object carrying
            ``corner_lo`` / ``corner_hi`` (``DesignRegion``,
            ``TopologyDesignRegion``). Corners resolve to cells the way
            ``topology_optimize`` resolves them — ``grid.position_to_index``
            on each corner, both ends INCLUSIVE — and the realized cell
            counts, not the declared extent, are what
            *design_eps_override* must match. One difference: a corner
            outside the interior is CLAMPED by ``topology_optimize`` and
            RAISES here, so a region that reaches into the absorber is a
            smaller design region there and an error on this path.

            What it buys: the Yee E update is linear in the fields, so the
            backward pass needs the primal field only where a coefficient is
            a design variable. With the grid-wide coefficients left constant
            and only this box recomputed from the traced permittivity,
            reverse-mode AD stores box-shaped arrays per timestep where
            ``eps_override`` stores grid-shaped ones. The gradient is the
            same gradient (measured to 2.5e-14 relative in float64 against
            ``eps_override`` on a 57x57x41 open-domain fixture); this is a
            change of formulation, not an approximation, and it needs
            neither ``checkpoint_segments`` nor any recompute.

            **Pass ``checkpoint=False`` with it.** ``checkpoint=True`` (this
            method's default) remats every step, which saves the scan CARRY
            — the six grid-sized field arrays — once per step no matter what
            the coefficients are. Measured on a 23x21x19 grid over 80 steps
            with a 3x3x3 box: 0.59 MB of residuals at ``checkpoint=False``
            against 26.2 MB at ``checkpoint=True``. The box removes the
            coefficient term; only turning the per-step remat off removes
            the carry term with it.

            The GRADED MESH carries it too (#1183): the corners resolve
            against the non-uniform grid's own cell edges and the box redo
            uses the graded curl, so a thin substrate meshed with a
            ``dz_profile`` gets the same box-shaped tape. On that lane a
            waveguide port, an MSL port and ``until_decay`` raise — the
            first two write their own field or ``materials`` across a whole
            plane from the background permittivity.

            Fenced, not degraded — for a design PERMITTIVITY
            (*design_eps_override*): ``distributed=True``,
            ``solver='adi'``, ``boundary='upml'``, Debye/Lorentz
            dispersion, Kerr, subpixel/anisotropic permittivity,
            ``pec_occupancy_override``, ``stencil_order=4``, the oblique
            Bloch path, combining with ``eps_override`` / ``sigma_override``
            / ``mu_r_override``, a box that reaches into the CPML absorber,
            and a box holding a source, port, lumped-RLC or
            surface-impedance-sheet cell all RAISE (a lumped or wire port
            may stay inside with *design_box_holds_ports*). Each of those either
            computes E by some other rule inside the box or reads the
            background permittivity at a box cell, and a silently dropped
            design variable is a zero gradient that reads like convergence.
            A design OCCUPANCY has its own, shorter list — see
            *design_occupancy_override*; in particular it does combine with
            ``eps_override`` and with ``pec_occupancy_override``, because
            it reads no material array.
        design_eps_override : jnp.ndarray or None
            Relative permittivity at the *design_box* cells, shaped like the
            realized box. Usually the traced quantity. Required with
            *design_box* unless *design_occupancy_override* is given
            instead, and rejected without a box.
        design_sigma_override : jnp.ndarray, 3-tuple of them, or None
            Conductivity at the *design_box* cells, same shape. ``None``
            (default) keeps the run's own conductivity there, so a lossy
            background inside the box is carried exactly rather than
            silently replaced by a lossless one. A 3-tuple
            ``(sigma_x, sigma_y, sigma_z)`` gives each E component its own
            conductivity — what a conducting SHEET needs, whose current runs
            along its two in-plane edges only. See
            :class:`~rfx.simulation.DesignBoxSpec`.
        design_occupancy_override : jnp.ndarray or None
            Issue #1183. Relaxed-conductor occupancy in ``[0, 1]`` at the
            *design_box* cells — metal-shape design (a notch stub, topology
            optimisation) rather than a dielectric one. It REPLACES
            ``pec_occupancy_override`` at those cells and carries it
            everywhere else, and the gradient is the same gradient
            ``pec_occupancy_override`` gives with the same values written
            into it.

            What it buys is the same trade: ``apply_pec_occupancy`` scales
            every E component by ``1 - M``, so a traced whole-grid occupancy
            makes the backward pass keep three GRID-sized primal E arrays
            per timestep. Confined to a box, the arrays it keeps cover the
            box plus one cell — the reach of the incident-cell rule — and
            ``1 - M`` itself is built once outside the time loop, because an
            occupancy is a design variable and not a field. Pass
            ``checkpoint=False`` with it, for the reason above.

            Uniform single-device Yee lane only; a graded mesh,
            ``distributed=True`` and ``solver='adi'`` raise. So do a
            periodic axis (the incident rule wraps across the seam, so the
            box's window is no longer the whole set of cells its occupancy
            can move), combining it with *design_eps_override* in one call,
            a box covering a cell the port setup CLEARS the occupancy at
            (the port cell, its six face neighbours, an MSL port's in-plane
            diagonal owners), and the Kottke occupancy lane
            (``RFX_PEC_OCC_KOTTKE=1``), where the occupancy becomes an
            inverse-eps tensor inside the E update that the box's values
            never reach.
        design_box_holds_ports : bool
            Let a permittivity *design_box* contain lumped and wire ports
            (default ``False``: a box holding a port raises). A port is a
            device on its edges — a 50 ohm load and a drive built from the
            drawn materials before the time loop — so each edge a port drives
            or loads inside the box's update window keeps the coefficients
            the drawn materials give it, for the update and the drive alike:
            the design values do not reach it, their derivative through it is
            exactly zero, and a box holding a port with the background
            written into it gives the run without the box. Holding keeps the
            DRAWN material in the port's feed gap, so the modelled structure
            differs from the same design written as a whole-grid
            ``eps_override`` by that gap. Passive ports (``excite=False``)
            are held too. The other edges of the port's cells stay design
            variables; at a port's cell a design value is the cell's volume
            material. The held edges come back as
            ``ForwardResult.design_box_held_edges``, ``(axis, i, j, k)`` with
            axis 0, 1, 2 for Ex, Ey, Ez. A soft source (even on a port's
            edge), a magnetic source and a lumped RLC element in the box
            still raise; an MSL port's termination is not held.
        pec_mask_override : jnp.ndarray or None
            Additional hard PEC mask to merge with geometry-defined PEC.
        pinned_sheet_layout : tuple or None
            Compile-once layout switch on the graded single-device lane. One
            entry per add_pinned_sheet call: None or two binary edge arrays in
            increasing tangential-axis order. For normal z their shapes are
            (i1-i0, j1-j0+1) and (i1-i0+1, j1-j0). Concrete nonbinary values
            and differentiation with respect to this layout are refused.
        pec_occupancy_override : jnp.ndarray or None
            Relaxed conductor occupancy field in ``[0, 1]`` for
            differentiable PEC-style optimisation.
        n_steps : int or None
            Number of timesteps. If None, auto-computed from *num_periods*.
        num_periods : float
            Number of periods at freq_max for auto step count.
        checkpoint : bool or None
            None enables checkpointing except on uniform distributed forward.
        checkpoint_segments : int or None
            If set, route the uniform single-device forward through the
            segmented-checkpoint path (issue #73), which trades ≈2x compute
            for ≈sqrt(n_steps) reverse-mode memory.  Must divide *n_steps*
            exactly (padding is rejected so DFT accumulator windows do not
            shift).  ``None`` (default) keeps the legacy per-step
            ``jax.checkpoint`` behaviour.  Currently wired only on the
            uniform single-device path; non-uniform meshes and
            ``distributed=True`` raise ``NotImplementedError`` (use
            *checkpoint_every* on the non-uniform path instead).
        emit_time_series : bool
            Emit the probe time series in the returned ``ForwardResult``
            (default True).  ``emit_time_series=False`` skips the time-series
            buffers and is currently only supported on the non-uniform
            forward path; on the uniform path it raises
            ``NotImplementedError`` (frequency-domain objectives such as NTFF
            and S-params still emit the series there).
        checkpoint_every : int or None
            Non-uniform-mesh counterpart of *checkpoint_segments* (a chunk
            size, not a segment count; issue #73).  When set to a positive
            integer ``K``, the non-uniform scan is run as a scan-of-scan that
            remats every ``K`` steps for ≈sqrt(n_steps) memory.  Currently
            only supported on the non-uniform forward path; on the uniform
            path it raises ``NotImplementedError`` (use *checkpoint_segments*
            there).  ``None`` (default) leaves forward-only runs unchanged.
        n_warmup : int
            Number of leading timesteps to run with the carry
            ``stop_gradient``'d so reverse-mode AD builds no tape for the
            initial transient (issue #40).  Only the trailing
            ``n_steps - n_warmup`` steps participate in autodiff.  Must
            satisfy ``n_warmup < n_steps``.  Default ``0`` (all steps
            differentiated).

            The truncation error DEPENDS ON DISTANCE FROM THE SOURCE TO THE
            DESIGN REGION — it is not simply "an approximation" that always
            costs accuracy (an earlier version of this docstring implied the
            error applies unconditionally; that was an overgeneralization
            from a single, worst-case fixture — see below). Severing the
            carry severs every gradient path through steps ``< n_warmup``,
            but only steps during which the WAVEFRONT HAS ALREADY REACHED
            the design region carry any such path to sever; before the
            wavefront arrives the field there is ~0, so severing those
            steps discards ~nothing and the gradient is (near-)exact.
            Compute a safe bound yourself::

                K_safe ~= floor(min_distance(source, design_region) / (C0 * dt))

            (``min_distance`` is the minimum over EVERY active source AND
            over each source's own spatial extent, not just its nominal
            position — a TFSF plane-wave source illuminates from an
            entire box face, not a point; grid steps for the wavefront to
            reach the closest design cell, ``C0`` = vacuum lightspeed,
            ``dt`` = ``sim._build_grid().dt`` /
            ``sim._build_nonuniform_grid().dt`` — a conservative floor valid
            for any slower propagation). ``n_warmup <= K_safe`` stays
            within this repo's ~1.5% AD-vs-FD noise floor (near-exact,
            <0.1%, well below ``K_safe``); well beyond ``K_safe``, error
            grows further and can reach the full gradient magnitude by
            the time ``n_warmup`` reaches the loss window. Two measured
            regimes (issue #626 part 2 / addendum, both vs an independent
            central-FD oracle): a NEAR-SOURCE placement (design cell ~3
            cells from the source, so ``K_safe ~ 0``) shows error growing
            smoothly from a ~1.5% noise floor up to 58% at the
            loss-window boundary itself, and EXACTLY zero (gradient fully
            vanished) once ``n_warmup`` extends far enough into the loss
            window (``tests/unit/autodiff/test_n_warmup.py::
            test_warmup_truncation_error_grows_with_k``); a
            FAR-FROM-SOURCE placement (design cell 62 cells from the
            source, ``K_safe=108`` measured from the grid's own ``dt``)
            shows AD matching the ``n_warmup=0`` FD oracle to <0.01%
            through ``K_safe``-20, then growing SMOOTHLY (not a sharp
            cliff) as ``n_warmup`` approaches ``K_safe`` -- 0.26% at
            ``K_safe``-4, 0.60% at ``K_safe``-2, 1.19% AT ``K_safe``
            itself (still inside the noise floor above) -- continuing to
            grow past it toward 75% by ``K_safe`` + 92 —
            ``scripts/diagnostics/i626_n_warmup_wavefront_locality.py``
            (finely sampled around ``K_safe``; an earlier version of this
            docstring, read from a coarser sweep that skipped every point
            near the boundary, overstated the transition as a sharp
            cliff "exactly at K_safe"). The near-source curve above is
            the WORST case, not the general
            case: use ``K_safe`` to decide whether YOUR source/design
            placement is in the exact or the truncated regime before
            assuming ``n_warmup`` costs accuracy. Forward (non-AD) output
            is exactly ``n_warmup``-invariant (bit-identical) in every
            placement measured; only the gradient is affected. Currently
            only supported on the non-uniform / distributed-non-uniform
            forward paths; on the uniform single-device lane it raises
            ``NotImplementedError`` (issue #626 — previously silently
            accepted and ignored, worse than an error).  Use
            *checkpoint_segments* for uniform-lane reverse-mode memory
            relief instead — it is exact regardless of placement, at the
            cost of recompute rather than gradient accuracy.
        skip_preflight : bool
            Skip the consolidated :meth:`preflight` validation suite that
            normally runs before the forward simulation (default False).
            Use only when preflight has already been run by the caller
            (e.g. :func:`rfx.optimize` runs it once at entry) or to bypass a
            known-spurious warning on a deliberate configuration.
        distributed : bool, optional
            **Opt-in, unstable, pending GPU evidence (issue #44).**  When
            ``True``, use the mesh's distributed runner across ``devices``.
            Uniform meshes share run(devices=...) including its one-device
            fallbacks; graded meshes use the distributed NU runner.
            Uniform distributed execution refuses material/design overrides
            and explicit checkpointing; an omitted checkpoint uses the
            runner's ordinary loop. Other lanes default to checkpointing.
        devices : list of jax.Device or None, optional
            Devices for the distributed run.  When ``distributed=True``
            and ``devices=None``, defaults to ``jax.devices()``.  When
            an explicit list is supplied, every device must already exist
            in ``jax.devices()`` (otherwise ``ValueError``).  Passing
            ``devices=`` *without* ``distributed=True`` raises
            ``ValueError`` — there is no silent activation of the
            distributed lane.
        exchange_interval : int, optional
            Ghost-cell exchange interval for the distributed runner
            (default ``1``).  Only ``1`` is currently supported by
            ``run_nonuniform_distributed_pec``; other values are
            forward-compatible reservations and will raise inside the
            runner.
        port_s11_freqs : array-like or None
            Frequencies (Hz) at which to accumulate per-port V/I DFTs inside
            the JIT scan body so that ``ForwardResult.s_params`` is populated
            with wave-decomposition |S11| values for lumped/wire ports
            (issue #72) — the AD-traceable counterpart of
            ``run(compute_s_params=True)``.  Required by the
            :func:`minimize_s11_at_freq_wave_decomp` objective.  Currently
            supported on uniform and graded single-device paths;
            graded distributed execution raises ``NotImplementedError``.
        rlc_values_override : dict or None
            Differentiable-value injection for lumped RLC elements added with
            :meth:`add_lumped_rlc` (WP 4-E).  ``LumpedRLCSpec`` stores plain
            floats, so a component value can only enter the AD tape as a tracer
            through this override.  Maps a 0-based element index (registration
            order over :meth:`add_lumped_rlc` calls) to a dict with any of the
            keys ``"R"`` / ``"L"`` / ``"C"`` whose values replace the
            registered floats — e.g.
            ``forward(rlc_values_override={0: {"R": R, "C": C}}, ...)`` makes R
            and C of the first RLC element differentiable.  A missing element
            index or key falls back to the registered spec float.  ``None``
            (default) uses the registered floats for every element.  Only wired
            on the uniform single-device path.  NOTE: on this path a registered
            lumped RLC element now correctly affects ``forward()`` even without
            an override — previously it was a silent no-op.
            Uniform-path only; the non-uniform / distributed lanes do not
            iterate ``self._lumped_rlc`` in ``forward()``.
        ringdown : rfx.ringdown.RingdownSpec or None
            Issue #1254. Complete the wire-port S-parameters of a record cut
            while the structure still rings, inside the differentiable
            program: ``ForwardResult.ringdown.s_params`` is the completion
            ``run(ringdown=...)`` gives of the same record (the poles of the
            port voltage and current identified on ``[window_start T, T]``,
            the unrecorded tail of each spectrum added in closed form), and
            ``jax.grad`` through it differentiates the infinite-record
            spectrum (the poles by the frozen Gauss-Newton implicit
            derivative, the residues re-fitted), not the cut record.
            ``.s_params_long`` is the completion from the window that starts
            twice as early; the gradient through it, against the gradient
            through ``.s_params``, is the gradient's witness
            (:func:`rfx.ringdown.gradient_witness`). ``.report`` holds the
            checks of the value (W0, W1, WE and the rest, as ``run()``'s),
            computed on the host from a concrete result and ``None`` while
            traced. Wire ports only (``add_port(..., extent=...)``); on the
            uniform lane one port is supported; ``port_s11_freqs=`` (the
            bins) is needed on both lanes. Everything ``run(ringdown=...)`` refuses is refused, and
            so are ``distributed=True`` and ``emit_time_series=False``.
            ``None`` (default) leaves every output and the traced program as
            they were.

        **_removed_kwargs
            Rejection shim for removed keywords, providing migration errors;
            it does not accept additional simulation options.

        Returns
        -------
        ForwardResult
            Minimal differentiable observables (time series and optional NTFF).
            ``settling_db`` and ``settling_witness`` score retained user-probe
            records on the concrete result using the run() diagnostic and
            underflow floor. They are host diagnostics, unavailable during
            tracing, and can be read on concrete records outside the jitted
            objective. Other result fields keep their existing JIT
            restrictions, including the host Grid object. Absent
            records yield None rather than a passing value. Use
            ``settling_verdict(result.settling_db)`` for the shared -40 dB bar.

        Notes
        -----
        ``forward()`` refuses refinement: it has no subgridded solve.
        ``forward()`` can be called inside ``jax.jit``, so an optimisation
        step compiles once: ``step = jax.jit(jax.value_and_grad(loss))``
        compiles on its first call and reuses the program afterwards, while
        an un-jitted ``jax.value_and_grad(loss)`` compiles the whole solve
        again on every call (#1225). Under an outer ``jax.jit`` (single
        device), the set-up built from the model alone is evaluated while
        tracing and the time stepping is compiled into the caller's program
        (#1367, #1354), so a function with no traced input compiles too. The
        set-up's arrays become constants of that program: the first call
        costs a plain call's set-up on top of the compile, and the compiled
        program holds those arrays for as long as JAX caches it. The loop
        compiles over operands XLA cannot read, as in a plain call; what is
        computed from a traced input before the loop (a port's drive from a
        traced permittivity, #1320) is compiled with the caller's program,
        so the jitted value and gradient agree with a plain call to float32
        rounding, not necessarily bit for bit.
        """
        self, pinned_sheet_layout = _conductors.prepare_sheet_layout(
            self, pinned_sheet_layout, distributed=distributed)
        refuse_forward_smoothing(self, subpixel_smoothing=subpixel_smoothing,
            eps_override=eps_override,
            pec_occupancy=pec_occupancy_override, design_occupancy=design_occupancy_override)
        if gradient not in ("autodiff", "adjoint"):
            raise ValueError("gradient must be 'autodiff' or 'adjoint'")
        if gradient == "adjoint":
            from rfx.adjoint import admit_forward_adjoint
            admit_forward_adjoint(
                self, distributed=distributed, ringdown=ringdown,
                design_box=design_box, design_eps=design_eps_override,
                design_sigma=design_sigma_override,
                other_overrides=(eps_override, sigma_override, mu_r_override,
                                 pec_mask_override, pec_occupancy_override,
                                 design_occupancy_override, rlc_values_override),
                port_s11_freqs=port_s11_freqs)
        if _removed_kwargs:
            _reject_removed_forward_kwargs(_removed_kwargs)
        validate_exchange_interval(exchange_interval)
        if ringdown is not None:
            from rfx.ringdown import refuse_forward_request
            refuse_forward_request(self, ringdown, distributed=distributed,
                                   emit_time_series=emit_time_series)

        # Phase 3 (issue #44 V3 §M6): one-shot UserWarning for the opt-in
        # distributed=True path so users know the path is opt-in / unstable
        # / pending GPU evidence.  The flag lives at module scope so we
        # warn exactly once per process, not per Simulation instance.
        global _DISTRIBUTED_FIRST_CALL_WARNED
        if distributed and not _DISTRIBUTED_FIRST_CALL_WARNED:
            import warnings as _w
            _w.warn(
                "Simulation.forward(distributed=True) is opt-in and pending "
                "GPU evidence (see issue #44). The distributed lane is "
                "'experimental / scaling' and not part of the "
                "correctness-bearing baseline; see the Distributed row of "
                "docs/guides/support_matrix.md before relying on it.",
                UserWarning,
                stacklevel=2,
            )
            _DISTRIBUTED_FIRST_CALL_WARNED = True

        # Phase 3 (V3 dispatch rule): devices= without distributed=True
        # must be rejected with a clear ValueError.  No silent activation.
        if devices is not None and not distributed:
            raise ValueError(
                "forward(devices=...) requires distributed=True; "
                "passing devices without distributed=True is rejected to "
                "avoid silent activation of the distributed lane. "
                "Either set distributed=True or omit devices."
            )

        if self._coaxial_ports:
            raise NotImplementedError(
                "add_coaxial_port() is not wired into Simulation.forward(). "
                "A coaxial port's S-parameters come from "
                "compute_coaxial_line_reflection(...) (one-port reflection) "
                "or compute_coaxial_two_port(...) (through line); both take "
                "eps_scale= for jax.grad. Use add_port(..., extent=...) for "
                "differentiable probe-feed S11 objectives."
            )

        if port_s11_freqs is not None or ringdown is not None:
            self._validate_forward_sparameter_request()
        _line_stub_scope = _line_stub_admit(self, port_s11_freqs)

        _solve_assembly = self._auto_preflight(skip=skip_preflight, context="forward", prepare=True, distributed=distributed, tfsf_material_overrides=(eps_override, sigma_override, mu_r_override))

        # ---- (2,4) stencil fence: reject order=4 on unsupported lanes ----
        self._check_stencil_order_supported(distributed=distributed)

        # ---- W6.3 unified lane dispatch: one place decides + rejects ----
        plan = self._dispatch_plan(
            mode="forward", subpixel_smoothing=subpixel_smoothing,
            diagnostics=_solve_assembly.diagnostics,
            n_steps=n_steps,
            num_periods=num_periods,
            distributed=distributed,
            port_s11_freqs=port_s11_freqs,
            checkpoint_segments=checkpoint_segments,
            emit_time_series=emit_time_series,
            checkpoint_every=checkpoint_every,
            n_warmup=n_warmup, tfsf_material_overrides=(eps_override, sigma_override, mu_r_override),
        )
        if _realized.ACTIVE is not None:
            _realized.enter(self, plan.lane)
        if plan.lane == "fwd_distributed":
            from rfx.runners._distributed_entry import forward_distributed
            return forward_distributed(
                self, plan=plan, assembly=_solve_assembly, devices=devices,
                exchange_interval=exchange_interval, num_periods=num_periods,
                port_s11_freqs=port_s11_freqs, skip_preflight=skip_preflight,
                eps_override=eps_override, sigma_override=sigma_override,
                mu_r_override=mu_r_override, pec_mask_override=pec_mask_override,
                pec_occupancy_override=pec_occupancy_override, design_box=design_box,
                design_eps_override=design_eps_override, design_sigma_override=design_sigma_override,
                design_occupancy_override=design_occupancy_override,
                design_box_holds_ports=design_box_holds_ports, rlc_values_override=rlc_values_override,
                subpixel_smoothing=subpixel_smoothing, checkpoint=checkpoint, checkpoint_every=checkpoint_every,
                checkpoint_segments=checkpoint_segments, n_warmup=n_warmup,
                emit_time_series=emit_time_series, ringdown=ringdown)
        checkpoint = True if checkpoint is None else checkpoint
        # No forward lane (and so no optimize()) applies Dey-Mittra weights.
        self._refuse_conformal_boundary(
            {"fwd_uniform": "uniform forward",
             "fwd_nonuniform": "non-uniform forward",
             "fwd_distributed_nu": "distributed non-uniform forward",
             }.get(plan.lane, plan.lane),
            entry="the differentiable forward solve (forward/optimize or an S-matrix override)",
            instead="use run() on a uniform mesh to keep conformal PEC "
                    "for a forward-only result")
        if ringdown is not None:
            from rfx.ringdown import refuse_forward_lane
            refuse_forward_lane(plan.lane, sum(
                1 for p in self._ports
                if float(p.impedance) > 0.0 and p.extent is not None),
                port_s11_freqs=port_s11_freqs)

        # WP 4-E: rlc_values_override is only wired on the uniform lane.  Fail
        # loudly rather than silently returning a zero gradient on a lane that
        # does not iterate self._lumped_rlc in forward().
        if rlc_values_override is not None and plan.lane != "fwd_uniform":
            raise NotImplementedError(
                "forward(rlc_values_override=...) is only supported on the "
                f"uniform single-device forward lane, not {plan.lane!r}. "
                "Lumped RLC component-value gradients require the uniform mesh "
                "path (non-uniform / distributed forward do not process "
                "add_lumped_rlc elements)."
            )

        # Reactive Kerr χ³ (#437) is threaded through the shared _run scan on the
        # UNIFORM forward lane only.  The non-uniform / distributed forward lanes
        # discard kerr_chi3, so a Kerr material there would silently give LINEAR
        # physics (and a linear gradient) — fail loud on those lanes only.
        if (plan.lane != "fwd_uniform"
                and any(getattr(m, "chi3", 0.0) != 0.0 for m in self._materials.values())):
            raise NotImplementedError(
                "forward() supports Kerr χ³ (nonlinear) materials only on the uniform "
                f"single-device lane, not {plan.lane!r} (which discards chi3 → silent "
                "linear physics + linear gradient). Use a uniform mesh + single device "
                "for differentiable nonlinear (Kerr) design."
            )

        # Both single-device lanes build TFSF; distributed/ADI lanes do not.
        if self._tfsf is not None and plan.lane not in ("fwd_uniform", "fwd_nonuniform"):
            raise NotImplementedError(
                "Differentiable TFSF plane-wave forward refuses distributed and "
                f"ADI lanes, including {plan.lane!r}. Use a uniform or graded "
                "single-device forward lane for TFSF/plane-wave inverse design."
            )

        # mu_r_override is wired only on the uniform lane; the NU/distributed
        # material-override paths do not thread it, so fail loud instead of
        # silently dropping it to a zero-gradient no-op.
        if mu_r_override is not None and self._solver == "adi":
            raise NotImplementedError(
                "forward(mu_r_override=...) is not supported with solver='adi': "
                "the ADI update does not read the permeability override. "
                "Use solver='yee' on a uniform single-device mesh, or omit mu_r_override."
            )
        if mu_r_override is not None and plan.lane != "fwd_uniform":
            raise NotImplementedError(
                "mu_r_override (differentiable permeability) is supported only on "
                f"the uniform single-device forward lane, not {plan.lane!r}."
            )

        # #1179/#1183 design box: the lanes that carry it. The distributed
        # and ADI runners never see ``design_box``, so without this they
        # would run the BACKGROUND design variable and return a zero
        # gradient — a silent wrong answer.
        _design_requested = (design_box is not None
                             or design_eps_override is not None
                             or design_sigma_override is not None
                             or design_occupancy_override is not None
                             or bool(design_box_holds_ports))
        if _design_requested:
            _design_lanes = (("fwd_uniform", "fwd_nonuniform")
                             if design_occupancy_override is None
                             else ("fwd_uniform",))
            if plan.lane not in _design_lanes:
                raise NotImplementedError(
                    "forward(design_box=...) is supported on the "
                    + " / ".join(_design_lanes)
                    + f" forward lane(s), not {plan.lane!r} "
                    "(#1179 permittivity, #1183 graded mesh and occupancy). "
                    "Use eps_override / pec_occupancy_override there.")
            if self._solver != "yee":
                raise NotImplementedError(
                    "forward(design_box=...) is supported only on the Yee "
                    f"solver, not solver={self._solver!r} (#1179): the ADI "
                    "lane builds its own implicit update coefficients from "
                    "``materials``. Use eps_override there.")

        if plan.lane == "fwd_distributed_nu" and self._interface_eps == "dual_average":
            raise ValueError("interface_eps='dual_average' cannot combine with forward eps_override or distributed NU")
        if plan.lane == "fwd_distributed_nu":
            from rfx.runners._distributed_nu_execute import dispatch_forward
            return dispatch_forward(self,
                plan=plan, subpixel_smoothing=subpixel_smoothing,
                _solve_assembly=_solve_assembly,
                eps_override=eps_override,
                sigma_override=sigma_override,
                pec_mask_override=pec_mask_override,
                pec_occupancy_override=pec_occupancy_override,
                checkpoint=checkpoint,
                emit_time_series=emit_time_series,
                checkpoint_every=checkpoint_every,
                n_warmup=n_warmup,
                devices=devices,
                exchange_interval=exchange_interval,
                skip_preflight=skip_preflight,
                port_s11_freqs=port_s11_freqs,
                num_periods=num_periods,
                design_sigma_override=design_sigma_override,
                design_eps_override=design_eps_override,
            )

        if plan.lane == "fwd_nonuniform":
            _nu_design_spec = None
            if _design_requested:
                # #1183: the corners resolve against the GRADED grid, whose
                # z cell edges are not a multiple of any dx. Everything the
                # resolved step context decides is
                # ``rfx.model.resolve_design_box``'s, from
                # ``_build_nu_scan``.
                _nu_design_spec, _ = self._resolve_design_box_override(
                    self._build_nonuniform_grid(),
                    design_box=design_box,
                    design_eps_override=design_eps_override,
                    design_sigma_override=design_sigma_override,
                    design_occupancy_override=design_occupancy_override,
                    eps_override=eps_override,
                    sigma_override=sigma_override,
                    mu_r_override=mu_r_override,
                    pec_occupancy_override=pec_occupancy_override,
                    debye_spec=None,
                    lorentz_spec=None,
                    kerr_chi3=None,
                    holds_ports=design_box_holds_ports,
                )
            _nu_fwd_call = functools.partial(
                self._forward_nonuniform_from_materials,
                subpixel_smoothing=subpixel_smoothing,
                port_s11_freqs=port_s11_freqs,
                eps_override=eps_override,
                sigma_override=sigma_override,
                pec_mask_override=pec_mask_override,
                pec_occupancy_override=pec_occupancy_override,
                pinned_sheet_layout=pinned_sheet_layout,
                design_box=_nu_design_spec,
                n_steps=plan.n_steps,
                checkpoint=checkpoint,
                emit_time_series=emit_time_series,
                checkpoint_every=checkpoint_every,
                n_warmup=n_warmup,
            )
            _nu_fwd_call, _nu_grid = _conductors.bind_solve_call(_solve_assembly, _nu_fwd_call)
            if ringdown is None:
                result = _nu_fwd_call()
            else:
                from rfx.ringdown import RingdownForward, host_bins
                result = RingdownForward(
                    self, ringdown, lane="graded", n_steps=plan.n_steps,
                    grid=_nu_grid, bins=None if port_s11_freqs is None else host_bins(port_s11_freqs, np.float32)).run(_nu_fwd_call)
            result = _conductors.with_sheet_layout_reports(result, pinned_sheet_layout)
            if _nu_design_spec is not None:
                result = result._replace(
                    design_box_held_edges=_nu_design_spec.held_edges)
            return with_conductivity_record(self._attach_run_settling_witness(
                result, n_steps=plan.n_steps, num_periods=num_periods, context="forward"),
                design_sigma=design_sigma_override, sigma_override=sigma_override, design_eps=design_eps_override, n_steps=plan.n_steps, dt=_nu_grid.dt)

        # ---- Uniform forward lane (plan.lane == "fwd_uniform") ----
        n_steps = plan.n_steps
        conductors, grid, assembly_inputs, _fwd_sheet_specs, _fwd_pec_sheets, _fwd_pec_wires = _conductors.uniform_solve_inputs(_solve_assembly)
        materials, debye_spec, lorentz_spec, pec_mask, pec_shapes, _, kerr_chi3 = assembly_inputs

        if eps_override is not None or sigma_override is not None or mu_r_override is not None:
            materials = apply_material_overrides(
                materials, eps_override=eps_override, sigma_override=sigma_override,
                mu_r_override=mu_r_override)

        if pec_mask_override is not None:
            pec_mask = pec_mask_override if pec_mask is None else (pec_mask | pec_mask_override)

        if n_steps is None:
            if self._solver == "adi":
                dt_adi = float(grid.dt * self._adi_cfl_factor)
                n_steps = steps_for_periods(dt_adi, grid.freq_max, num_periods)
            else:
                n_steps = grid.num_timesteps(num_periods=num_periods)

        # #677: node-thin sheet ctx against the realized PEC edges of this
        # forward run (PEC wins on overlapping edges).  #931: the PEC
        # sheets AND WIRES belong in that realization too — they own no
        # cell, so the "is there any PEC?" question cannot be asked of
        # ``pec_mask`` and the sheet list alone.  A model whose only
        # conductor is a filament used to reach the lossy operator with
        # ``pec_edge_masks=None``, and the operator then wrote field back
        # onto the wire's own PEC edge.
        #
        # ``periodic=`` on the OUTER call is the same #689 requirement:
        # the builder realizes the f0 footprint's own edges, and on a
        # periodic axis the seam edge (node n-1 to node 0) is in the sheet.
        # Without the flags it is zero-padded away, so the seam carries no
        # loss — measured on an x-periodic f0 sheet at dx = 2 mm, 35 loaded
        # Ex edges through run() against 30 through forward().
        # ``self._periodic_flags()`` is the single spelling (#931 §1.7).
        _fwd_sheet_ctx = _conductors.forward_sheet_context(conductors, pec_mask)
        # #679: the same UPML refusal run_uniform carries. forward() reaches
        # the solver by its own route (it never enters run_uniform), so
        # WITHOUT this the eps_override / forward() channel silently ran the
        # combination run() refuses: _forward_from_materials passes
        # boundary='upml' straight through, simulation.py applies
        # apply_upml_e and then lets the sheet operator overwrite that
        # split-field E at its edges. Gated on self._boundary alone, exactly
        # like run_uniform, so the two channels accept the same configs.
        if _fwd_sheet_ctx is not None and self._boundary == "upml":
            raise ValueError(
                "surface-impedance (surface_impedance_f0) sheets are not "
                "supported with boundary='upml' on the uniform forward() / "
                "eps_override lane (#677 v1): UPML replaces the whole E "
                "update with its split-field form, which the sheet operator "
                "would silently override at its edges. Use boundary='cpml' "
                "or 'pec'.")
        if _fwd_sheet_ctx is not None and (
                debye_spec is not None or lorentz_spec is not None):
            raise ValueError(
                "surface-impedance (surface_impedance_f0) sheets combined "
                "with dispersive (Debye/Lorentz) materials in one run are "
                "not supported (#677 v1): the sheet operator would "
                "silently override the ADE dispersion update at its edges.")

        _design_spec = None
        _design_occ_spec = None
        if _design_requested:
            _design_spec, _design_occ_spec = self._resolve_design_box_override(
                grid,
                design_box=design_box,
                design_eps_override=design_eps_override,
                design_sigma_override=design_sigma_override,
                design_occupancy_override=design_occupancy_override,
                eps_override=eps_override,
                sigma_override=sigma_override,
                mu_r_override=mu_r_override,
                pec_occupancy_override=pec_occupancy_override,
                debye_spec=debye_spec,
                lorentz_spec=lorentz_spec,
                kerr_chi3=kerr_chi3,
                holds_ports=design_box_holds_ports,
            )
        if _design_spec is not None:
            # The design permittivity sets the precision of the material
            # arithmetic, exactly as an ``eps_override`` array does — that
            # path replaces ``eps_r`` outright, so a float64 override makes
            # the whole grid's coefficients float64. Without this the two
            # halves of one grid would be stepped at two precisions: the
            # float32 rounding of ``eps_r * EPS_0`` alone moves the field by
            # ~1e-7 relative, which is 1e9 x the difference between the two
            # formulations. ``_assemble_materials`` pins float32 (#646
            # promote-never-pin), so the promotion belongs here.
            materials = apply_material_overrides(materials, design_box=_design_spec)

        _fwd_call = functools.partial(
            self._forward_from_materials,
            grid,
            materials,
            debye_spec,
            lorentz_spec,
            subpixel_smoothing=subpixel_smoothing,
            conductors=conductors,
            n_steps=n_steps,
            checkpoint=checkpoint,
            checkpoint_segments=checkpoint_segments,
            pec_mask=pec_mask, pec_shapes=pec_shapes,
            pec_sheets=tuple(_fwd_pec_sheets),
            pec_wires=tuple(_fwd_pec_wires),
            pec_occupancy=pec_occupancy_override,
            kerr_chi3=kerr_chi3,
            port_s11_freqs=port_s11_freqs,
            rlc_values_override=rlc_values_override,
            sheet_impedance=_fwd_sheet_ctx,
            design_box=_design_spec,
            gradient=gradient,
            design_occupancy=_design_occ_spec,
            monitor_overrides={
                "eps_r": eps_override,
                "sigma": sigma_override,
                "mu_r": mu_r_override,
                "pec_mask": pec_mask_override,
                "pec_occupancy": pec_occupancy_override,
            },
            lane="fwd_uniform",
            diagnostics=_solve_assembly.diagnostics,
        )
        if ringdown is None:
            _res = _fwd_call()
        else:
            from rfx.ringdown import RingdownForward
            _res = RingdownForward(self, ringdown, lane="uniform", n_steps=n_steps,
                                   grid=grid, bins=port_s11_freqs).run(_fwd_call)
        if _design_spec is not None:
            _res = _res._replace(design_box_held_edges=_design_spec.held_edges)
        _warn_if_nonfinite_result(_res, context="forward")
        from rfx.current_moments import require_accumulated_current_moments
        require_accumulated_current_moments(self, _res, "forward")
        return with_conductivity_record(self._attach_run_settling_witness(_res, n_steps=n_steps, num_periods=num_periods, context="forward"),
            design_sigma=design_sigma_override, sigma_override=sigma_override, design_eps=design_eps_override, n_steps=n_steps, dt=getattr(_res, "dt", grid.dt))

    # ---- run ----

    def run(
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
        """Run the simulation.

        Parameters
        ----------
        n_steps : int or None
            Number of timesteps. If None, auto-computed from num_periods.
        num_periods : float
            Number of periods at freq_max for auto timestep count.
        checkpoint : bool
            Enable gradient checkpointing for reverse-mode AD.
        compute_s_params : bool or None
            Compute S-parameter matrix. Default: True when ports exist.
        s_param_freqs : array or None
            Frequencies for S-parameters. Default: 50 points from
            freq_max/10 to freq_max.
        s_param_n_steps : int or None
            Timesteps for S-parameter simulation (may need more than
            the main simulation for frequency resolution).
        until_decay : float or None
            When provided, overrides *n_steps*. On absorbing boundaries the
            run stops when the stored field energy left after the source has
            ended has decayed to this fraction of its value after the drive:
            the leapfrog-conserved interior energy
            ``W = sum((eps E^n.E^n + (mu0/eps0) mu H^(n-1/2).H^(n+1/2)) dV)``,
            checked every ``decay_check_interval`` steps from the drive end on
            against its running maximum over those checks. On a resonant
            structure the run is as long as the threshold asks. On a high-Q
            structure whose mode reaches the absorber, the stop can move by
            about 170 steps with the check phase while the spectra stay within
            1e-2 of the band peak. Closed uniform boxes keep the instantaneous
            point-field stop (see :func:`rfx.simulation.run_until_decay`).
            **Non-uniform meshes (issue #383):** the same rule on absorbing
            boundaries (per-cell dV); closed boundaries refuse ``until_decay``.
            On the decay path every step-sized buffer is allocated at
            ``decay_max_steps``; NU flux monitors keep the rectangular window.
            In a lossy dielectric the run can last longer than the field's
            ringing: charge moved onto the dielectric's surfaces during the
            drive relaxes with the time constant eps/sigma after it. That field
            is quasi-static (out of band), the record stays correct, and the
            stop record and the cap advisory say what ended the run.
            **Float32 floor:** a check also counts as below when W is under
            ``KAPPA = 1e-12`` of the largest W at any check (drive included)
            and has stopped falling (under 5 % decay per 50 steps). Risk: a mode
            below 1e-12 of the drive's peak energy that decays more slowly than
            5 % per 50 steps is stopped as rounding. Every run adds one
            ``until_decay.stop`` Diagnostic to ``Result.diagnostics``
            (criterion ``threshold``, ``float32_floor`` or ``cap``).
            **DC (issue #388):** a soft source whose waveform has relative
            DC above 1e-6 leaves a static charge that holds W up; ``run()``
            warns and names it (``rfx.measurement.decay``).
        decay_check_interval : int
            Check decay every N steps (default 50).
        decay_min_steps : int
            Always run at least this many steps (default 100).
        decay_max_steps : int
            Hard upper limit on steps (default 50000).
        decay_energy_consecutive : int
            Absorbing-boundary only: consecutive sub-threshold energy checks
            before stopping (default ``2``; values below 2 are refused).
        radiated_flux_box : tuple or None
            Physical lower/upper corners of a closed box enclosing the radiator,
            clear of CPML. Selects outgoing-flux decay instead of interior-energy
            decay on absorbing boundaries; None keeps the energy criterion.
        flux_env_checks : int
            Number of recent checks whose maximum absolute flux forms the
            radiated-flux envelope (default 4).
        snapshot : SnapshotSpec or None
            Field snapshot schedule and selection; None disables snapshots.
        subpixel_smoothing : bool or str
            False: staircase; True: diagonal, first order on curved interfaces (worst at eps ~30+).
            "full_tensor": full anisotropic averaging, uniform single-device; synthetic eps-4 sphere extinction measured second order over three meshes.
            Farjadpour 2006; Oskooi 2009; Werner-Bauer-Cary 2013 node form. "kottke_pec": unified PEC occupancy rule.
        skip_preflight : bool
            Skip advisory preflight checks. Runtime admission guards still apply.
        decay_monitor_component : str
            Field component to monitor (default ``"ez"``). Used only by the
            closed/PEC point-field fallback stop.
        decay_monitor_position : tuple or None
            Physical position to monitor. If None, use domain center.
        conformal_pec : bool
            Enable Dey-Mittra conformal PEC for second-order accuracy
            on curved PEC surfaces. Default False (staircase PEC).
        conformal_min_weight : float
            Minimum conformal weight for CFL stability clamping.
            Default 0.1. Recommended range: 0.05-0.3.
        devices : list of jax.Device or None
            When a list with len > 1 is provided, run the simulation
            distributed across those devices using 1D slab decomposition
            along the x-axis via ``shard_map``. Supported combinations are
            listed in ``docs/guides/support_matrix.md``.
        exchange_interval : int, optional
            Ghost exchange interval in timesteps; only integer 1 is supported.
        report_every : int or None
            Issue #667 — progress reporting for long solves. When set,
            print one ``  [PROGRESS] ...`` line every *N* timesteps giving
            steps done / total, wall elapsed, implied rate and ETA, on the
            same stdout channel as preflight so a redirected job log
            captures it. ``None`` (default) is OFF and leaves the solve on
            its unchanged code path, so nothing moves for existing callers.

            Supported on the **uniform lane only** (both the fixed
            ``n_steps`` scan and ``until_decay``); the distributed,
            non-uniform, ADI and subgridded lanes warn and drop it rather
            than silently producing no output. It is forward-only and
            raises under ``jax.jit``/``grad``/``vmap`` (the check inspects
            the carry, the per-step inputs and the material / geometry
            arrays — the routes ``forward()``/``optimize()`` trace through).

            On the fixed-``n_steps`` scan the reporting turns the single
            compiled scan into ``ceil(n_steps / report_every)`` calls to
            the SAME compiled scan with ``carry`` threaded through — a
            continuation, not a re-solve, and bit-exact against
            ``report_every=None`` (locked by
            ``tests/unit/runners/test_run_progress_reporting.py``).

            Each report costs one device synchronisation plus one print.
            On a 118k-cell CPU fixture that cost was **not resolvable above
            the measurement noise** at any cadence tried: a paired
            off-vs-off self-control read -4.8% [-9.3, -0.9], and every
            cadence from 2000 down to 25 landed inside that floor. The one
            clean figure is +0.7-0.8% for a single-chunk run, bounding the
            fixed per-report cost. So choose ``report_every`` for a useful
            reporting interval, not to minimise an overhead this benchmark
            could not detect; only the direction (finer cadence costs more)
            is established. None of this speaks to the large-grid GPU
            regime, which was not measured.
        report_label : str
            Short tag prefixed to each progress line (e.g. ``"drive p1"``).
            Ignored when ``report_every`` is None.
        ringdown : rfx.ringdown.RingdownSpec or None
            Issue #1254 — complete the wire-port S-parameters of a record
            cut while the structure still rings. The slow poles are
            identified on ``[window_start * T, T]`` of the port voltage and
            current (the source must be off there) and the unrecorded tail
            of each spectrum is added in closed form. The result is
            ``Result.ringdown`` (``.s_params`` on the bins of
            ``Result.s_params``, ``.report`` with the window, the poles and
            the witnesses: ``WE`` when available (otherwise ``W2``), no growing
            pole, passivity, source off). Every other output is the one the
            same run gives without ``ringdown=``. Wire ports
            (``add_port(..., extent=...)``) on the uniform (one port) and
            graded-mesh lanes, fixed record length; other ports, TFSF, Kerr,
            ``devices=`` and ``until_decay`` are refused with the reason.
        until_identified : bool
            Issue #1254 — with ``ringdown=``, end the run once the completion
            is witnessed instead of at ``n_steps``, which becomes the longest
            record allowed. The record is checked on a geometric schedule
            (every check at least 1.25x the record of the last); the run
            stops at the first check where every source is off over
            ``[T/4, T]``, the error witness ``WE`` is within ``witness_tol``
            there and at the check before, the record is at least half the
            amplitude decay time ``Q / (pi f)`` of the slowest mode whose own
            unrecorded tail moves the completed S by ``witness_tol`` (a static
            field left in a closed box is left out; the largest such decay
            time over the checks the two-in-a-row rule rests on), and the
            completion reads passive with no growing pole. The result is the
            one ``run(n_steps=T, ringdown=...)`` returns for the stopping
            record ``T``, with the stop report (every check's record, ``WE``,
            the pole that set the floor, the conditions and the check's wall
            time) in ``Result.ringdown.stop``. Each check identifies the
            record twice (0.1-3 s on a CPU for the test cavity); on a small
            model that costs more than the steps it saves. A weakly coupled
            high-Q mode holds the floor for half its own decay time, which
            can be longer than ``n_steps``. When no check
            stops the run, the whole ``n_steps`` record is completed and the
            report names the condition that failed. Refused without
            ``ringdown=``, with ``until_decay``, ``snapshot=``, DFT planes
            and flux monitors.

        Returns
        -------
        Result
        """
        # The caller's arguments as given, before anything below resolves them:
        # a devices= model that runs on one device is re-run with all of them.
        self._resolve_field_dtype()
        _call_args = dict(locals())
        del _call_args["self"]
        admit_traced_run(self, until_decay, decay_max_steps)
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

        if self._coaxial_ports:
            raise NotImplementedError(
                "add_coaxial_port() is not wired into Simulation.run(). A "
                "coaxial port's S-parameters come from "
                "compute_coaxial_line_reflection(...) (one-port reflection) "
                "or compute_coaxial_two_port(...) (through line); for a "
                "single add_coaxial_port(), compute_s_matrix(method=...) "
                "dispatches to either. Use add_port(..., extent=...) for "
                "probe-feed S-parameters."
            )

        self._validate_run_sparameter_request(
            compute_s_params=compute_s_params,
            s_param_freqs=s_param_freqs,
            s_param_n_steps=s_param_n_steps,
            devices=devices,
        )
        _line_stub_scope = _line_stub_admit(self, s_param_freqs)

        _distributed_run = devices is not None and len(devices) > 1
        _solve_assembly = self._auto_preflight(skip=skip_preflight, context="run", check_ntff="advisory", prepare=True, distributed=_distributed_run)
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
                _diagnostics=_diagnostics, _call_args=_call_args,
                n_steps=n_steps, num_periods=num_periods, devices=devices,
                exchange_interval=exchange_interval, skip_preflight=skip_preflight,
                compute_s_params=compute_s_params, s_param_freqs=s_param_freqs,
                s_param_n_steps=s_param_n_steps, conformal_pec=conformal_pec,
                subpixel_smoothing=subpixel_smoothing, checkpoint=checkpoint,
                snapshot=snapshot, until_decay=until_decay, report_every=report_every)

        # ---- Non-uniform mesh lane ----
        if plan.lane == "run_nonuniform":
            # #383: until_decay is supported on the NU lane for absorbing
            # boundaries (the interior-energy criterion, same class as the
            # uniform #169 stop). Closed/PEC NU domains refuse it with a
            # lane-accurate reason: their interior energy does not decay,
            # and the NU lane has no point-field fallback. The NU lane's
            # S-parameters come from the main run's port record, so an
            # s_param_n_steps other than that record's length is refused.
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
                        "non-uniform lane has no point-field fallback "
                        "(issue #383)",
                    "s_param_n_steps":
                        "this lane reads the S-parameters from the main "
                        "run's port record, whose length is n_steps",
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
            _nu_call, _nu_grid = _conductors.bind_solve_call(_solve_assembly, _nu_call)
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
        from rfx.realized_geometry import attach_record
        geometry_record = _conductors.early_run_record(self, conductors, plan.lane)

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
                    "sheets or wires (#931): the coarse/fine SBP-SAT "
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
