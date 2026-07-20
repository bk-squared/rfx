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

import math
import os
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from rfx.grid import Grid
from rfx.core.yee import MaterialArrays
from rfx.geometry.csg import Box  # noqa: F401  (referenced by moved docstrings/comments)
from rfx.sources.sources import GaussianPulse  # noqa: F401  (local import in moved bodies)
from rfx.materials.debye import init_debye  # noqa: F401  (local import in moved bodies)
from rfx.materials.lorentz import init_lorentz  # noqa: F401  (local import in moved bodies)
from rfx.adi import ADIState2D, run_adi_2d
from rfx.boundaries.spec import BoundarySpec  # noqa: F401  (referenced by moved comments)
from rfx.simulation import SnapshotSpec  # noqa: F401  (run() signature type-hint)
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


class _ExecuteMixin:
    """Execute cluster mixin: forward / run dispatch + per-path runners.

    Mixed into ``Simulation``; all methods stay bound methods on a
    ``Simulation`` instance (resolved via MRO).
    """

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
        missed. order=2 (the default) is unaffected.
        """
        if getattr(self, "_stencil_order", 2) != 4:
            return
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
        compute_s_params=None,
        s_param_freqs=None,
        s_param_n_steps=None,
    ):
        """Run simulation using SBP-SAT subgridding (JIT-compiled)."""
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
            s_param_freqs=s_param_freqs,
            s_param_n_steps=s_param_n_steps,
        )

    # ---- non-uniform mesh run path ----


    def _run_nonuniform(self, *, n_steps, compute_s_params=None,
                        s_param_freqs=None,
                        subpixel_smoothing: bool | str = False,
                        checkpoint: bool = False,
                        until_decay: float | None = None,
                        decay_check_interval: int = 50,
                        decay_min_steps: int = 100,
                        decay_max_steps: int = 50_000,
                        decay_energy_consecutive: int = 2):
        """Run simulation on non-uniform grid with graded dz.

        ``until_decay`` (issue #383) is threaded through to
        ``run_nonuniform_path`` only when the caller has already gated on
        absorbing boundaries — ``run()`` passes ``None`` (warn-and-drop)
        for closed-boundary NU sims, where the interior energy does not
        decay. ``decay_monitor_component`` / ``decay_monitor_position``
        are NOT threaded: they parameterize the uniform lane's
        closed/PEC point-field fallback, which has no NU counterpart.
        """
        # Defense-in-depth: the NU runner does not honour stencil_order.
        self._check_stencil_order_supported()
        self._reject_refplane_ports_off_uniform_lane("non-uniform mesh",
                                                     compute_s_params)
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
            n_steps=n_steps,
            compute_s_params=compute_s_params,
            s_param_freqs=s_param_freqs,
            subpixel_smoothing=subpixel_smoothing,
            checkpoint=checkpoint,
            until_decay=until_decay,
            decay_check_interval=decay_check_interval,
            decay_min_steps=decay_min_steps,
            decay_max_steps=decay_max_steps,
            decay_energy_consecutive=decay_energy_consecutive,
        )

    @staticmethod
    def _warn_run_sparams_if_nonpassive(result) -> None:
        """Apply the shared lumped/wire advisory to an assembled run result."""
        s_params = getattr(result, "s_params", None)
        freqs = getattr(result, "freqs", None)
        if s_params is None or freqs is None:
            return
        from rfx.probes.probes import warn_if_nonpassive_lumped_s11
        warn_if_nonpassive_lumped_s11(
            s_params,
            freqs,
            extractor="run(compute_s_params=True)",
        )

    def _warn_postrun_energy_witness(
        self,
        result,
        *,
        fixed_num_periods: bool,
        until_decay: float | None,
    ) -> None:
        """Warn about measured late energy or an exactly zero excitation."""
        import warnings

        time_series = getattr(result, "time_series", None)
        series = None if time_series is None else np.asarray(time_series)
        has_probe_series = series is not None and series.size > 0

        if has_probe_series:
            magnitude = np.abs(series)
            peak = float(np.max(magnitude))
            if peak == 0.0:
                warnings.warn(
                    "no field energy was recorded — the source may not have "
                    "coupled (position off-grid, wrong component, or zero "
                    "amplitude) (issue #336)",
                    stacklevel=3,
                )
                return

            pulsed = any(
                isinstance(getattr(source, "waveform", None), GaussianPulse)
                for source in (*self._ports, *self._msl_ports)
            )
            open_boundary = self._boundary_spec.absorber_type in ("cpml", "upml")
            if open_boundary and pulsed and fixed_num_periods and until_decay is None:
                tail_samples = max(1, math.ceil(magnitude.shape[0] * 0.05))
                tail = float(np.max(magnitude[-tail_samples:]))
                tail_db = -math.inf if tail == 0.0 else 20.0 * math.log10(tail / peak)
                if tail_db > -40.0:
                    warnings.warn(
                        f"run ended at {tail_db:.1f} dB of peak — ring-down "
                        "truncated; Harminv/DFT/NTFF quantities may carry "
                        "transient error; consider until_decay or more periods "
                        "(issue #332)",
                        stacklevel=3,
                    )
            return

        state = getattr(result, "state", None)
        fields = [
            np.asarray(field)
            for name in ("ex", "ey", "ez", "hx", "hy", "hz")
            if (field := getattr(state, name, None)) is not None
        ]
        if fields and all(field.size > 0 and float(np.max(np.abs(field))) == 0.0
                          for field in fields):
            warnings.warn(
                "no field energy was recorded — the source may not have "
                "coupled (position off-grid, wrong component, or zero "
                "amplitude) (issue #336)",
                stacklevel=3,
            )

    def _warn_until_decay_dc_floor(self, *, dt, n_table: int) -> None:
        """#388 predictor: warn when a soft source will floor ``until_decay``.

        A soft current source deposits its waveform's discrete time integral
        as a net static end-charge ``q = S/((1+loss)*dz_src)`` with
        ``S = sum s(t_n)*dt`` — measured exactly on both the uniform and
        non-uniform lanes (issue #388). The remnant's energy is the
        discrete-Poisson self-energy of that charge: it does not decay and
        CPML cannot absorb a static field, so it floors the interior-energy
        decay criterion and ``until_decay`` cap-hits instead of stopping.

        For each soft source (``impedance == 0`` port entries, plus MSL
        ports — the same soft-Ez deposit mechanism drives them), compute
        ``rel_DC = |sum s(t_n)*dt| / sum |s(t_n)|*dt`` over the planned
        table length (``decay_max_steps`` — the decay lanes' buffer size)
        and warn quantitatively above 1e-3 (the demo-scale GaussianPulse
        sits at ~6e-5; a bandwidth=1.2 ModulatedGaussian at ~0.68).

        The message is TRUNCATION-AWARE: rel_DC over the planned table
        conflates intrinsic waveform DC with table-truncation DC. When the
        table ends inside the source pulse (``n_table*dt`` below the
        pulse-completion time ``t_done = t0 + 3*tau``), the estimate is
        truncation-dominated — the remedy is a longer ``decay_max_steps``
        (or a fixed ``n_steps``), and raising the waveform ``cutoff``
        makes it WORSE because it lengthens the onset (measured: default
        pulse + decay_max_steps=200 → rel_DC 2.14e-2 pure truncation
        artifact; cutoff=4.5 at n_table~500 fires while cutoff=3 is
        silent). Only past pulse completion do the intrinsic-DC remedies
        (higher cutoff / narrower ModulatedGaussian) apply.

        Lives with the run()-path warning helpers (not preflight) because
        ``preflight()`` does not receive run kwargs and this advisory is
        meaningful only when ``until_decay`` is requested.
        """
        import warnings

        from rfx.core.jax_utils import is_tracer

        if is_tracer(dt):
            return  # traced dt: advisory host math is impossible here
        dt = float(dt)
        if not dt > 0.0 or n_table <= 0:
            return
        t = np.arange(int(n_table), dtype=np.float64) * dt
        soft_entries = [
            pe for pe in self._ports
            if pe.impedance == 0.0 and getattr(pe, "excite", True)
        ]
        # MSL ports launch through the same soft-Ez deposit across the
        # trace cross-section, so their waveform DC floors the criterion
        # identically. Entry access mirrors the sibling #386
        # unresolved-pulse validator: read the ``waveform`` attribute,
        # skip string-named waveforms (below).
        soft_entries += [
            pe for pe in self._msl_ports if getattr(pe, "excite", True)
        ]
        for pe in soft_entries:
            wf = getattr(pe, "waveform", None)
            if wf is None or isinstance(wf, str):
                continue
            try:
                s = np.asarray(wf(jnp.asarray(t)), dtype=np.float64)
            except Exception:
                continue  # non-vectorizable CustomWaveform etc. — skip
            if s.shape != t.shape or not np.all(np.isfinite(s)):
                continue
            denom = float(np.sum(np.abs(s))) * dt
            if denom == 0.0:
                continue
            s_int = float(np.sum(s)) * dt
            rel_dc = abs(s_int) / denom
            if rel_dc <= 1e-3:
                continue
            # Pulse-completion time t_done = t0 + 3*tau (the waveform's
            # own t0 property when present, 3*tau otherwise). A table
            # ending before t_done makes rel_DC truncation-dominated.
            _loc = f"{pe.position} ({getattr(pe, 'component', 'msl port')})"
            t_done = None
            try:
                _tau = float(wf.tau)
            except (AttributeError, TypeError, ValueError):
                _tau = None
            if _tau is not None and math.isfinite(_tau) and _tau > 0.0:
                try:
                    _t0 = float(wf.t0)
                except (AttributeError, TypeError, ValueError):
                    _t0 = 3.0 * _tau
                t_done = _t0 + 3.0 * _tau
            if t_done is not None and n_table * dt < t_done:
                warnings.warn(
                    f"soft source at {_loc}: waveform {type(wf).__name__} "
                    f"shows relative DC {rel_dc:.2e} over the planned "
                    f"{int(n_table)}-step table, but the planned "
                    f"decay_max_steps window ends inside the source pulse "
                    f"(table {n_table * dt:.3e} s < pulse completion "
                    f"t0+3*tau ~ {t_done:.3e} s) — the DC estimate is "
                    "truncation-dominated, not intrinsic waveform DC; "
                    "until_decay may cap-hit (issue #388). Increase "
                    "decay_max_steps (or run with a fixed n_steps) so the "
                    "table covers the full pulse; raising the waveform "
                    "cutoff makes this WORSE (it lengthens the onset).",
                    UserWarning, stacklevel=3,
                )
            else:
                warnings.warn(
                    f"soft source at {_loc}: "
                    f"waveform {type(wf).__name__} deposits relative DC "
                    f"{rel_dc:.2e} (|sum s*dt| = {abs(s_int):.3e} over the "
                    f"planned {int(n_table)}-step table) — the deposited "
                    "static charge will floor the interior-energy criterion "
                    "(a static field neither decays nor is absorbed by "
                    "CPML); until_decay may cap-hit (issue #388). Reduce "
                    "the waveform's DC content: a higher GaussianPulse "
                    "cutoff (e.g. cutoff=4.5), ModulatedGaussian with "
                    "bandwidth <= 0.4, or run with a fixed n_steps. Note: "
                    "this advisory does NOT catch low-rel_DC floors "
                    "amplified by thin source cells (the #388 patch class "
                    "measured the floor at rel_DC ~6e-5) — see issue #388.",
                    UserWarning, stacklevel=3,
                )

    def _warn_decay_param_sanity(self, *, until_decay, decay_min_steps,
                                 decay_max_steps) -> None:
        """Behaviour-NEUTRAL sanity advisories on the decay-stop params.

        Warn-only, single ``run()``-level site so both lanes that honour
        ``until_decay`` (uniform and absorbing-boundary non-uniform) are
        covered by one check (post-#392 review):

        * ``decay_min_steps > decay_max_steps`` — every check index lies
          past the table end, so zero energy checks occur and the run is
          a forced ``decay_max_steps`` run (the stop can never fire).
        * ``until_decay >= 1`` — the criterion ``U < until_decay*peak_U``
          is satisfied while the field energy is still rising, so the
          stop can fire immediately after ``decay_min_steps``.
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
                "(U < until_decay * peak_U) is met while the field energy "
                "is still rising, so the stop can fire immediately after "
                "decay_min_steps; use a fraction well below 1 (e.g. 1e-3).",
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
    def _warn_unsupported_run_kwargs(path_name: str,
                                     unsupported_kwargs: dict,
                                     reason_overrides: dict | None = None) -> None:
        """Emit ``UserWarning`` for any Simulation.run kwarg that a given
        dispatch path drops. Only non-default values are surfaced.

        Previously the distributed / non-uniform / subgridded paths
        silently dropped most of the ``run`` kwargs (fixed 2026-04); this helper makes
        the drop explicit at the API boundary so users can tell their
        request was not honoured. See GitHub tracking issue for the
        feature-request backlog to actually propagate these kwargs.

        ``reason_overrides`` lets a caller replace the shared per-kwarg
        reason string with a lane-accurate one (e.g. the NU lane drops
        ``until_decay`` only on closed boundaries, for a different reason
        than the distributed/subgridded lanes drop it).
        """
        import warnings as _w
        # Per-kwarg "silent" values — the kwarg is dropped only if the
        # user set it to a value that asks the dispatch path to do
        # something it does not support. ``compute_s_params=False``
        # is silent because it matches the path's actual behaviour
        # (no S-matrix assembly), while ``compute_s_params=True``
        # warns because the user asked for something that will not
        # happen.
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
                "per-component anisotropic eps is not wired on this path",
            "checkpoint":
                "reverse-mode AD will store the full tape (no "
                "checkpoint-every support here)",
            "snapshot":
                "scan-body field snapshotting is not wired on this path",
            "until_decay":
                "scan body runs for exactly n_steps; decay-based "
                "termination is supported on the uniform lane and the "
                "absorbing-boundary (cpml/upml) non-uniform lane only",
            "conformal_pec":
                "Dey-Mittra conformal weights are computed on a uniform "
                "staircase mesh only",
            "compute_s_params":
                "S-matrix assembly is not plumbed through this path",
            "s_param_freqs":
                "S-matrix assembly is not plumbed through this path",
            "s_param_n_steps":
                "S-matrix assembly is not plumbed through this path",
        }
        if reason_overrides:
            reasons.update(reason_overrides)
        for kw, val in unsupported_kwargs.items():
            silent = silent_values.get(kw, (None,))
            if val in silent:
                continue
            reason = reasons.get(kw, "not propagated")
            _w.warn(
                f"{kw}={val!r} is silently ignored on the {path_name} run "
                f"path ({reason}).",
                UserWarning, stacklevel=3,
            )

    def _port_sparameter_entries(self) -> list:
        """Return ``add_port`` entries that are actual impedance ports.

        ``add_source`` reuses ``_PortEntry`` with ``impedance=0`` as a
        source-only sentinel, so S-parameter request validation must not
        count every ``self._ports`` entry as a calibrated port.
        """

        return [pe for pe in self._ports if pe.impedance > 0.0]

    def _auto_configure_mesh(self) -> None:
        """P1: Auto-detect features and set dx/dz_profile when dx=None.

        Uses the existing auto_configure() infrastructure to derive cell size
        from geometry dimensions and material properties.  Runs only once per
        simulation — subsequent calls are no-ops.
        """
        import warnings as _w
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
        )

        self._dx = config.dx
        if config.dz_profile is not None and self._dz_profile is None:
            self._dz_profile = config.dz_profile
            # Update domain z from dz_profile
            dz_total = float(np.sum(config.dz_profile))
            self._domain = (self._domain[0], self._domain[1], dz_total)

        _w.warn(
            f"Auto mesh: dx={config.dx*1e3:.3f}mm "
            f"({config.cells_per_wavelength:.0f} cells/λ)"
            + (f", non-uniform z ({len(config.dz_profile)} cells)"
               if config.dz_profile is not None else "")
            + ". Set dx= explicitly to suppress.",
            stacklevel=3,
        )
        for w in config.warnings:
            _w.warn(w, stacklevel=3)

    def _auto_preflight(
        self, *, skip: bool = False, context: str = "forward",
        check_ntff: bool | str = True,
    ) -> None:
        """Emit a UserWarning if preflight finds issues (issue #66).

        Called automatically at the start of ``forward()``, ``optimize()``,
        and ``topology_optimize()`` so users discover physics violations
        (under-resolved mesh, geometry in CPML, probe in PEC, ...) before
        spending minutes of GPU compute. Pass ``skip_preflight=True`` at
        the call site to opt out (tests, already-validated configs).

        ``check_ntff`` gates the NTFF check family. ``run()`` passes
        ``check_ntff="advisory"`` (issue #303): the λ/4 near-field gap
        and sub-wavelength ground-plane (issue #334) advisories are
        physics-relevant to any far-field computation and
        run() is the common NTFF entry point, so they are surfaced as
        warnings — but only ``forward(port_s11_freqs=...)`` / ``optimize``
        (the inverse-design entry points) hard-fail on the PEC-overlap
        error. Previously run() skipped the family entirely and still
        printed "All checks passed", which contradicted explicit
        ``sim.preflight()`` output on the same configuration.
        """
        if skip:
            return
        # preflight(strict=False) COLLECTS findings as issues; it only raises if
        # a validator itself crashes (a bug, e.g. a non-ValueError). Let that
        # propagate loudly (Phase D) — do NOT degrade a validator bug to a soft
        # warning that hides it and lets a broken run proceed.
        issues = self.preflight(strict=False, check_ntff=check_ntff)
        if not issues:
            return
        import warnings
        errors = [i for i in issues if getattr(i, "severity", "warning") == "error"]
        warns = [i for i in issues if getattr(i, "severity", "warning") != "error"]
        if warns:
            body = "\n  - ".join(warns)
            warnings.warn(
                f"[{context}] preflight found {len(warns)} advisory issue(s) - "
                f"pass skip_preflight=True to suppress:\n  - {body}",
                UserWarning, stacklevel=3,
            )
        if errors:
            # Error-severity findings are structurally-impossible configs
            # (e.g. upml+refinement, Floquet+non-uniform-z) whose validators
            # raise ValueError. run()/forward() used to call those validators
            # directly so the error PROPAGATED; routing through preflight must
            # preserve that hard-fail, else a known-invalid run silently
            # proceeds. Re-raise (aligns with Tidy3D/Meep raise-on-setup-error).
            # skip_preflight=True remains the explicit escape hatch.
            detail = "\n  - ".join(errors)
            raise ValueError(
                f"[{context}] preflight found {len(errors)} blocking error(s) "
                f"(pass skip_preflight=True to bypass):\n  - {detail}"
            )

    def _run_adi_from_materials(
        self,
        grid: Grid,
        materials: MaterialArrays,
        debye_spec,
        lorentz_spec,
        *,
        n_steps: int,
        pec_mask: jnp.ndarray | None = None,
        return_state: bool = True,
    ):
        """Run the integrated ADI solver path (2D TMz or 3D)."""
        import copy

        self._validate_adi_configuration(materials, debye_spec, lorentz_spec)

        dt = float(grid.dt * self._adi_cfl_factor)
        times = jnp.arange(n_steps, dtype=jnp.float32) * dt

        grid_out = copy.copy(grid)
        grid_out.dt = dt

        # ---- 3D path ----
        if self._mode == "3d":
            from rfx.adi import run_adi_3d, ADIState3D, make_adi_absorbing_sigma_3d

            sources_3d = []
            for pe in self._ports:
                i, j, k = grid.position_to_index(pe.position)
                waveform = jax.vmap(pe.waveform)(times)
                sources_3d.append((i, j, k, pe.component, waveform))

            probes_3d = []
            for pe in self._probes:
                i, j, k = grid.position_to_index(pe.position)
                probes_3d.append((i, j, k, pe.component))

            eps_r_3d = materials.eps_r
            sigma_3d = materials.sigma

            if self._boundary == "cpml" and self._cpml_layers > 0:
                nx, ny, nz = grid.shape
                absorb_sigma = make_adi_absorbing_sigma_3d(
                    nx, ny, nz, self._cpml_layers, grid.dx, grid.dx, grid.dx)
                sigma_3d = sigma_3d + absorb_sigma

            shape = grid.shape
            zeros = jnp.zeros(shape, dtype=jnp.float32)
            ex_f, ey_f, ez_f, hx_f, hy_f, hz_f, probe_data = run_adi_3d(
                zeros, zeros, zeros, zeros, zeros, zeros,
                eps_r_3d, sigma_3d,
                dt, grid.dx, grid.dx, grid.dx,
                n_steps,
                sources=sources_3d,
                probes=probes_3d,
                pec_mask=pec_mask,
            )
            if probe_data is None:
                probe_data = jnp.zeros((n_steps, 0), dtype=jnp.float32)

            if return_state:
                state = ADIState3D(
                    ex=ex_f, ey=ey_f, ez=ez_f,
                    hx=hx_f, hy=hy_f, hz=hz_f,
                    step=jnp.asarray(n_steps, dtype=jnp.int32),
                )
                return Result(
                    state=state,
                    time_series=probe_data,
                    s_params=None, freqs=None,
                    grid=grid_out, dt=dt,
                    freq_range=(self._freq_max / 10, self._freq_max, self._boundary),
                )
            return ForwardResult(
                time_series=probe_data,
                ntff_data=None, ntff_box=None,
                grid=grid_out,
            )

        # ---- 2D TMz path ----
        sources = []
        for pe in self._ports:
            i, j, _ = grid.position_to_index(pe.position)
            waveform = jax.vmap(pe.waveform)(times)
            sources.append((i, j, waveform))

        probes = []
        for pe in self._probes:
            i, j, _ = grid.position_to_index(pe.position)
            probes.append((i, j, pe.component))

        eps_r_2d = materials.eps_r[:, :, 0]
        sigma_2d = materials.sigma[:, :, 0]

        # Add implicit absorbing sigma layer for CPML boundary
        if self._boundary == "cpml" and self._cpml_layers > 0:
            from rfx.adi import make_adi_absorbing_sigma
            nx_2d, ny_2d = eps_r_2d.shape
            absorb_sigma = make_adi_absorbing_sigma(
                nx_2d, ny_2d, self._cpml_layers, grid.dx)
            sigma_2d = sigma_2d + absorb_sigma

        pec_mask_2d = None
        if pec_mask is not None:
            pec_mask_2d = pec_mask[:, :, 0]
        ez0 = jnp.zeros_like(eps_r_2d)
        hx0 = jnp.zeros_like(eps_r_2d)
        hy0 = jnp.zeros_like(eps_r_2d)
        ez_f, hx_f, hy_f, probe_data = run_adi_2d(
            ez0,
            hx0,
            hy0,
            eps_r_2d,
            sigma_2d,
            dt,
            grid.dx,
            grid.dx,
            n_steps,
            sources=sources,
            probes=probes,
            pec_mask=pec_mask_2d,
        )
        if probe_data is None:
            probe_data = jnp.zeros((n_steps, 0), dtype=ez_f.dtype)

        if return_state:
            state = ADIState2D(
                ez=ez_f,
                hx=hx_f,
                hy=hy_f,
                step=jnp.asarray(n_steps, dtype=jnp.int32),
            )
            return Result(
                state=state,
                time_series=probe_data,
                s_params=None,
                freqs=None,
                grid=grid_out,
                dt=dt,
                freq_range=(self._freq_max / 10, self._freq_max, self._boundary),
            )

        return ForwardResult(
            time_series=probe_data,
            ntff_data=None,
            ntff_box=None,
            grid=grid_out,
        )

    def _forward_from_materials(
        self,
        grid: Grid,
        materials: MaterialArrays,
        debye_spec: tuple | None,
        lorentz_spec: tuple | None,
        *,
        n_steps: int,
        checkpoint: bool = True,
        checkpoint_segments: int | None = None,
        pec_mask: jnp.ndarray | None = None,
        pec_occupancy: jnp.ndarray | None = None,
        port_s11_freqs: object | None = None,
        rlc_values_override: dict | None = None,
        _sparam_drive_idx: int | None = None,
        _return_raw_port_sparams: bool = False,
    ) -> ForwardResult | dict:
        """Run a minimal differentiable forward path from explicit materials.

        Internal multi-drive S-matrix hook (item-5 Stage 1, 2026-06-22)
        --------------------------------------------------------------
        ``_sparam_drive_idx`` and ``_return_raw_port_sparams`` exist solely
        for the production-scan S-matrix driver
        (``rfx.probes.sparam_driver``).  Their defaults preserve the current
        ``forward()`` / ``run()`` behavior EXACTLY:

        * ``_sparam_drive_idx`` (default ``None``): when set to an integer,
          build the excitation source for ONLY the sparam-eligible
          lumped/wire port at that 0-based index (in registration order over
          the lumped+wire ports), ignoring every port's ``excite`` flag.  All
          other ports remain matched loads (their impedance is still folded
          into materials via ``setup_lumped_port`` / ``setup_wire_port``).
          When ``None``, excitation follows ``pe.excite`` as before.
        * ``_return_raw_port_sparams`` (default ``False``): when ``True``,
          return ``{"lumped": result.lumped_port_sparams,
          "wire": result.wire_port_sparams, "freqs": ...}`` (the all-port
          ``(v_dft, i_dft)`` accumulators) instead of the diagonal
          ``ForwardResult``.  When ``False``, the normal ``ForwardResult`` is
          returned unchanged.
        """
        if self._solver == "adi":
            return self._run_adi_from_materials(
                grid,
                materials,
                debye_spec,
                lorentz_spec,
                n_steps=n_steps,
                pec_mask=pec_mask,
                return_state=False,
            )

        from rfx.simulation import (
            run as _run,
            make_probe,
            make_port_source,
            make_wire_port_sources,
            LumpedPortSParamSpec,
            WirePortSParamSpec,
        )
        from rfx.sources.sources import (
            LumpedPort,
            WirePort,
            setup_lumped_port,
            setup_wire_port,
            _wire_port_cells,
        )

        # ── Lumped RLC elements — differentiable lane (WP 4-E) ──────────
        # Fold R/C into the jnp materials arrays FIRST (mirrors the concrete
        # run() ordering in runners/uniform.py:82), so subsequent port folds
        # stack on top. The traced metas are built at the very end (after all
        # material folding) just before the _run(...) call.  Component values
        # may be supplied AS tracers via ``rlc_values_override`` so
        # ``jax.grad`` flows w.r.t. R/L/C (LumpedRLCSpec stores plain floats).
        # For a sim with NO lumped RLC this block is skipped and ``rlc_metas``
        # stays None -> the _run(...) call is byte-identical to before.
        rlc_metas = None
        if self._lumped_rlc:
            from rfx.lumped import (
                setup_rlc_materials_traced,
                build_rlc_meta_traced,
            )
            _rlc_ov = rlc_values_override or {}
            for _rlc_idx, _rlc_spec in enumerate(self._lumped_rlc):
                _v = _rlc_ov.get(_rlc_idx, {})
                materials = setup_rlc_materials_traced(
                    grid, _rlc_spec, materials,
                    r_val=_v.get("R"), c_val=_v.get("C"),
                )

        sources = []
        probes = []
        pec_mask_local = pec_mask
        pec_occupancy_local = pec_occupancy
        lumped_port_sparam_specs: list = []
        wire_port_sparam_specs: list = []
        wire_refplane_specs: list = []
        # Resolve a freq array once for downstream auto-build (issue #72)
        if port_s11_freqs is not None:
            _s11_freqs_arr = jnp.asarray(port_s11_freqs, dtype=jnp.float32)
        else:
            _s11_freqs_arr = None

        # Collect all port cell indices for Kottke dilation guard (issue #82).
        _port_cleared_cells: list[tuple[int, int, int]] = []

        # Multi-drive S-matrix hook (item-5 Stage 1): 0-based counter over the
        # sparam-eligible lumped/wire ports (impedance != 0) in registration
        # order. When ``_sparam_drive_idx`` is set, only the port whose counter
        # equals it is excited; all others are matched loads regardless of
        # ``pe.excite``. When ``_sparam_drive_idx`` is None this counter is
        # inert and excitation follows ``pe.excite`` (current behavior).
        _sparam_port_idx = -1

        def _excite_this_port(pe) -> bool:
            if _sparam_drive_idx is None:
                return pe.excite
            return _sparam_port_idx == _sparam_drive_idx

        for pe in self._ports:
            if pe.impedance == 0.0:
                from rfx.simulation import make_j_source
                sources.append(
                    make_j_source(grid, pe.position, pe.component,
                                  pe.waveform, n_steps, materials)
                )
                continue

            # Sparam-eligible lumped/wire port — advance the multi-drive index.
            _sparam_port_idx += 1

            # Effective drive decision + waveform for this pass (issue #322).
            # A passive port (excite=False) registers waveform=None BY DESIGN
            # (add_port only defaults the waveform when excite=True). When the
            # S-matrix scan driver (_sparam_drive_idx; PR #258 wire reroute /
            # #214 lumped reroute) selects such a port as the driven one,
            # synthesize the same default excitation add_port() applies for
            # excite=True ports — otherwise make_port_source /
            # make_wire_port_sources hits jax.vmap(None)
            # ("TypeError: Expected a callable value, got None").
            _drive_this_port = _excite_this_port(pe)
            _drive_waveform = pe.waveform
            if _drive_this_port and _drive_waveform is None:
                _drive_waveform = GaussianPulse(
                    f0=self._freq_max / 2, bandwidth=0.8)

            if pe.extent is not None:
                axis_map = {"ex": 0, "ey": 1, "ez": 2}
                axis = axis_map[pe.component]
                end = list(pe.position)
                end[axis] += pe.extent
                wp = WirePort(
                    start=pe.position,
                    end=tuple(end),
                    component=pe.component,
                    impedance=pe.impedance,
                    excitation=_drive_waveform,
                )
                # Live-cell-aware fold + injection (issue #318): dead
                # extent cells inside PEC carry no port sigma and no
                # source. ``pec_mask_local`` at this point still holds the
                # assembled-geometry state for THIS port's cells (its own
                # clearing happens just below), which is exactly the
                # "live" definition.
                materials = setup_wire_port(grid, wp, materials,
                                            pec_mask=pec_mask_local)
                if _drive_this_port:
                    sources.extend(make_wire_port_sources(
                        grid, wp, materials, n_steps,
                        pec_mask=pec_mask_local))
                wp_cells = _wire_port_cells(grid, wp)
                # Clear PEC mask/occupancy at LIVE wire cells only (issue
                # #318 commit 2). Dead extent cells stay PEC — the old
                # all-cells clearing punched a one-cell in-plane
                # conductivity hole in the DUT conductor. The Kottke
                # occupancy-neighbor clearing below keys off
                # ``_port_cleared_cells`` and is scoped the same way.
                from rfx.sources.sources import _wire_port_live_cells
                _, wp_live_flags, _ = _wire_port_live_cells(
                    grid, wp, pec_mask_local)
                for cell, _live in zip(wp_cells, wp_live_flags):
                    if not _live:
                        continue
                    if pec_mask_local is not None:
                        pec_mask_local = pec_mask_local.at[cell[0], cell[1], cell[2]].set(False)
                    if pec_occupancy_local is not None:
                        pec_occupancy_local = pec_occupancy_local.at[cell[0], cell[1], cell[2]].set(0.0)
                    _port_cleared_cells.append((int(cell[0]), int(cell[1]), int(cell[2])))
                # Register a JIT-integrated S-param accumulator for this
                # WirePort when forward(port_s11_freqs=...) was requested
                # (issue #79 follow-up to PR #72). Mirrors the lumped
                # registration below; uses the wire's midpoint cell as the
                # V/I reference plane (consistent with wire_sparam_meta in
                # the JIT scan body of rfx/simulation.py).
                if _s11_freqs_arr is not None and wp_cells:
                    mid_cell = wp_cells[len(wp_cells) // 2]
                    wire_port_sparam_specs.append(WirePortSParamSpec(
                        mid_i=int(mid_cell[0]),
                        mid_j=int(mid_cell[1]),
                        mid_k=int(mid_cell[2]),
                        component=pe.component,
                        freqs=_s11_freqs_arr,
                        impedance=float(pe.impedance),
                    ))
                # Opt-in reference-plane V/I accumulators (issue #313).
                # Registered ONLY on the production-scan S-matrix driver
                # path (_sparam_drive_idx is not None): the plane waves
                # feed the OFF-diagonal S_ij and need the multi-drive
                # sweep; forward()'s diagonal path is untouched (its S11
                # never uses the planes), and the geometry derivation
                # below needs concrete (non-traced) pec_mask values.
                if (pe.reference_plane_cells is not None
                        and _s11_freqs_arr is not None
                        and _sparam_drive_idx is not None
                        and wp_cells):
                    from rfx.probes.refplane import build_wire_refplane_specs
                    # Guard: the planes must not reach past another port
                    # along the same line axis (they must sit on the
                    # uniform line BETWEEN the ports).
                    _axis_of = {"x": 0, "y": 1, "z": 2}
                    _l_ax = _axis_of[pe.direction[1]]
                    _sign = -1 if pe.direction[0] == "+" else +1
                    _i_port = int(grid.position_to_index(pe.position)[_l_ax])
                    _i_far = _i_port + _sign * 2 * int(pe.reference_plane_cells)
                    for _other in self._ports:
                        if _other is pe or _other.impedance == 0.0:
                            continue
                        _oi = int(grid.position_to_index(
                            _other.position)[_l_ax])
                        _in_zone = (_i_port < _oi <= _i_far) if _sign > 0 \
                            else (_i_far <= _oi < _i_port)
                        if _in_zone:
                            raise ValueError(
                                "reference_plane_cells: the reference "
                                f"planes of the port at {pe.position} "
                                f"(indices up to {_i_far} on axis {_l_ax}) "
                                "reach past another port at "
                                f"{_other.position} (index {_oi}). Reduce "
                                "N so both planes stay on the uniform "
                                "line between the ports. Note: this check "
                                "compares line-axis indices only "
                                "(conservative), so a TRANSVERSELY "
                                "separated port on a different parallel "
                                "trace also trips it — the check can be "
                                "revisited if that layout is needed.")
                    wire_refplane_specs.extend(build_wire_refplane_specs(
                        grid=grid,
                        port_cells=wp_cells,
                        e_component=pe.component,
                        impedance=float(pe.impedance),
                        direction=pe.direction,
                        n_cells=int(pe.reference_plane_cells),
                        freqs=_s11_freqs_arr,
                        port_index=_sparam_port_idx,
                        pec_mask=pec_mask,
                    ))
                continue

            lp = LumpedPort(
                position=pe.position,
                component=pe.component,
                impedance=pe.impedance,
                excitation=_drive_waveform,
            )
            materials = setup_lumped_port(grid, lp, materials)
            if _drive_this_port:
                sources.append(make_port_source(grid, lp, materials, n_steps))
            idx = grid.position_to_index(pe.position)
            if pec_mask_local is not None:
                pec_mask_local = pec_mask_local.at[idx[0], idx[1], idx[2]].set(False)
            if pec_occupancy_local is not None:
                pec_occupancy_local = pec_occupancy_local.at[idx[0], idx[1], idx[2]].set(0.0)
            _port_cleared_cells.append((int(idx[0]), int(idx[1]), int(idx[2])))
            # Register a JIT-integrated S-param accumulator for this
            # lumped port when the user requested forward(port_s11_freqs=...)
            # (issue #72). Skipping passive ports keeps S11 indexing
            # aligned with the active-port list.
            if _s11_freqs_arr is not None:
                lumped_port_sparam_specs.append(LumpedPortSParamSpec(
                    i=int(idx[0]), j=int(idx[1]), k=int(idx[2]),
                    component=pe.component,
                    freqs=_s11_freqs_arr,
                    impedance=float(pe.impedance),
                ))

        # MSL ports — full cross-section distributed feed; built like the
        # wire-port branch above but with N_y × N_z cells.  S-parameters are
        # extracted post-scan by ``compute_msl_s_matrix`` from registered
        # DFT plane probes (no JIT accumulator wiring needed here).
        if self._msl_ports:
            from rfx.sources.msl_port import (
                MSLPort,
                _msl_yz_cells,
                compute_msl_mode_profile,
                make_msl_port_sources,
                setup_msl_port,
            )
            for pe in self._msl_ports:
                x_feed, y_centre, z_lo = pe.position
                mp = MSLPort(
                    feed_x=float(x_feed),
                    y_lo=float(y_centre - pe.width / 2),
                    y_hi=float(y_centre + pe.width / 2),
                    z_lo=float(z_lo),
                    z_hi=float(z_lo + pe.height),
                    direction=pe.direction,
                    impedance=pe.impedance,
                    excitation=pe.waveform,
                )
                # Honour `pe.mode` so the source distribution matches
                # the imperative `run_uniform_path` (Phase 3 of gap #2/#4
                # closure, 2026-05-07).  ``laplace`` is the default for
                # ``add_msl_port`` and gives a structured Ez source from
                # the static-Laplace quasi-TEM mode shape.  ``eigenmode``
                # mode requires the J+M Schelkunoff source pair which the
                # legacy uniform forward path does not yet wire — refuse
                # explicitly so users do not silently get a wrong source.
                port_mode = getattr(pe, "mode", "uniform")
                mode_profile = None
                if port_mode == "laplace":
                    cells = _msl_yz_cells(grid, mp)
                    j_set = sorted({c[1] for c in cells})
                    k_set = sorted({c[2] for c in cells})
                    j_centre = (j_set[0] + j_set[-1]) // 2
                    k_mid = (k_set[0] + k_set[-1]) // 2
                    i_feed = cells[0][0]
                    if pe.eps_r_sub is not None:
                        eps_r_sub = float(pe.eps_r_sub)
                    else:
                        # G-AD-WIRE: materials.eps_r may be a JAX tracer
                        # when forward(eps_override=...) is active.
                        # The mode profile is a STATIC geometry quantity;
                        # use stop_gradient so np.asarray() never sees a
                        # tracer.  This does NOT break the AD tape for
                        # the DFT accumulators — the source distribution
                        # shape is fixed; only the FDTD field values
                        # (and hence the DFT plane accumulators) carry
                        # the gradient w.r.t. eps_override.
                        eps_r_sub = float(np.asarray(
                            jax.lax.stop_gradient(
                                materials.eps_r[i_feed, j_centre, k_mid]
                            )
                        ))
                    mode_profile = compute_msl_mode_profile(grid, mp, eps_r_sub)
                elif port_mode == "eigenmode":
                    raise NotImplementedError(
                        "MSL port mode='eigenmode' is supported on the "
                        "imperative Simulation.run() path "
                        "(`runners/uniform.py::run_uniform_path`) but not "
                        "yet on the differentiable Simulation.forward() "
                        "path — the J+M Schelkunoff source pair needs a "
                        "magnetic-source channel that `_forward_from_materials` "
                        "does not currently expose.  Use mode='laplace' "
                        "(the default) on the differentiable path."
                    )
                materials = setup_msl_port(grid, mp, materials,
                                           mode_profile=mode_profile)
                if pe.excite and pe.waveform is not None:
                    sources.extend(make_msl_port_sources(
                        grid, mp, materials, n_steps,
                        mode_profile=mode_profile,
                    ))
                # Clear PEC mask over the cross-section so the source/σ cells
                # are not zeroed by the PEC update.
                for cell in _msl_yz_cells(grid, mp):
                    if pec_mask_local is not None:
                        pec_mask_local = pec_mask_local.at[cell[0], cell[1], cell[2]].set(False)
                    if pec_occupancy_local is not None:
                        pec_occupancy_local = pec_occupancy_local.at[cell[0], cell[1], cell[2]].set(0.0)
                    _port_cleared_cells.append((int(cell[0]), int(cell[1]), int(cell[2])))

        for pe in self._probes:
            probes.append(make_probe(grid, pe.position, pe.component))

        if not probes and self._ports:
            for pe in self._ports:
                probes.append(make_probe(grid, pe.position, pe.component))

        _, debye, lorentz = self._init_dispersion(
            materials, grid.dt, debye_spec, lorentz_spec,
        )

        ntff_box = None
        if self._ntff is not None:
            from rfx.farfield import make_ntff_box
            corner_lo, corner_hi, freqs = self._ntff
            ntff_box = make_ntff_box(grid, corner_lo, corner_hi, freqs)

        # DFT plane probes — mirror runners/uniform.py:340-359 so the
        # JIT scan body actually accumulates plane-resolved DFT, then
        # carry the result back through ForwardResult.dft_planes for
        # plane-integrated V/I objectives (known gap #4, 2026-05-05).
        dft_planes = []
        if self._dft_planes:
            from rfx.probes.probes import init_dft_plane_probe
            _axis_to_index = {"x": 0, "y": 1, "z": 2}
            for pe in self._dft_planes:
                axis_idx = _axis_to_index[pe.axis]
                plane_pos = [0.0, 0.0, 0.0]
                plane_pos[axis_idx] = pe.coordinate
                grid_index = grid.position_to_index(tuple(plane_pos))[axis_idx]
                freqs_arr = (
                    pe.freqs if pe.freqs is not None
                    else jnp.linspace(self._freq_max / 10,
                                      self._freq_max, pe.n_freqs)
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

        # Waveguide ports (differentiable DFT accumulation inside scan)
        waveguide_ports = []
        if self._waveguide_ports:
            wg_freqs = None
            for pe in self._waveguide_ports:
                if pe.freqs is not None:
                    wg_freqs = jnp.asarray(pe.freqs, dtype=jnp.float32)
                    break
            if wg_freqs is None:
                wg_freqs = jnp.linspace(
                    self._freq_max * 0.5, self._freq_max, 20, dtype=jnp.float32)
            for pe in self._waveguide_ports:
                waveguide_ports.append(
                    self._build_waveguide_port_config(pe, grid, wg_freqs, n_steps))

        # Floquet ports — inject soft source, same as run_uniform.py:274-327
        periodic = None
        if self._periodic_axes:
            periodic = tuple(axis in self._periodic_axes for axis in "xyz")

        if self._floquet_ports:
            axis_map_str = {"x": 0, "y": 1, "z": 2}
            for fpe in self._floquet_ports:
                axis_idx = axis_map_str[fpe.axis]
                fp_f0 = fpe.f0 if fpe.f0 is not None else self._freq_max / 2
                from rfx.sources.sources import GaussianPulse as _GP
                wf = _GP(f0=fp_f0, bandwidth=fpe.bandwidth, amplitude=fpe.amplitude)
                center = [self._domain[i] / 2.0 for i in range(3)]
                center[axis_idx] = fpe.position
                if fpe.polarization == "te":
                    comp = {"z": "ex", "x": "ey", "y": "ex"}[fpe.axis]
                else:
                    comp = {"z": "ey", "x": "ez", "y": "ez"}[fpe.axis]
                from rfx.simulation import make_source as _make_src
                sources.append(_make_src(grid, tuple(center), comp, wf, n_steps))
            if periodic is None:
                periodic = (True, True, False)  # default x-y periodic for Floquet

        periodic_bool = periodic if periodic is not None else (False, False, False)

        # Forward cpml_axes from the grid — when waveguide ports are
        # present the grid restricts CPML to the non-propagation axes.
        # The default _run cpml_axes="xyz" builds CPML state for axes
        # that have no padding, producing shape-broadcast errors like
        # (8,1,1) vs (nx,ny,nz) during the scan (issue #29). The run()
        # path forwards these explicitly (see Simulation.run in _execute.py),
        # so does the waveguide compute path (see _sparams.py).
        cpml_axes_run = grid.cpml_axes
        pec_axes_run = "".join(a for a in "xyz" if a not in cpml_axes_run)

        # Stage 2 Kottke for AD-traceable PEC density (opt-in via env
        # ``RFX_PEC_OCC_KOTTKE=1``).  When enabled and
        # ── Port-aware Kottke dilation guard (issue #82) ──────────
        # Kottke's ``occ_dilated = max(f, roll(f,±1,...))`` picks up
        # non-zero occupancy from cells adjacent to each port cell.
        # For a probe-fed patch (port 1 cell below the PEC sheet),
        # the z+1 neighbor carries the patch occupancy into the port's
        # inv_eps, causing 60x–223% AD gradient mismatch.  Fix: also
        # zero the 6 face-neighbors of every port cell before Kottke.
        if pec_occupancy_local is not None and _port_cleared_cells:
            for ci, cj, ck in _port_cleared_cells:
                for di, dj, dk in ((1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)):
                    ni = min(max(ci + di, 0), grid.nx - 1)
                    nj = min(max(cj + dj, 0), grid.ny - 1)
                    nk = min(max(ck + dk, 0), grid.nz - 1)
                    pec_occupancy_local = pec_occupancy_local.at[ni, nj, nk].set(0.0)

        # ``pec_occupancy_local`` is supplied, build an ``aniso_inv_eps``
        # tensor via Kottke's PEC limit ((1−f)/ε perpendicular, 0
        # parallel) using the gradient of the occupancy as the
        # interface normal.  This routes the override through the same
        # subpixel machinery hard ``Box(material="pec")`` uses
        # (``compute_inv_eps_tensor_diag``), eliminating the sub-β
        # wiggle near high-Q resonances.  Bypasses the legacy
        # ``apply_pec_occupancy`` E-zeroing path on this branch — the
        # two would double-correct at sigmoid edges (cf. Stage 2
        # design memo §R9 anti-pattern).
        aniso_inv_eps_run = None
        pec_occupancy_for_run = pec_occupancy_local
        if (pec_occupancy_local is not None and
                os.environ.get("RFX_PEC_OCC_KOTTKE", "0") not in ("0", "", "false", "False")):
            from rfx.geometry.smoothing import kottke_inv_eps_from_occupancy
            inv_baseline = (
                (1.0 / materials.eps_r).astype(jnp.float32),
                (1.0 / materials.eps_r).astype(jnp.float32),
                (1.0 / materials.eps_r).astype(jnp.float32),
            )
            aniso_inv_eps_run = kottke_inv_eps_from_occupancy(
                grid,
                pec_occupancy_local,
                aniso_inv_eps_baseline=inv_baseline,
            )
            pec_occupancy_for_run = None
            if os.environ.get("RFX_PEC_OCC_KOTTKE_DEBUG", "0") not in ("0", "", "false", "False"):
                import sys as _sys
                _ix, _iy, _iz = aniso_inv_eps_run
                print(f"[kottke debug] occ shape={pec_occupancy_local.shape} "
                      f"min={float(jnp.min(pec_occupancy_local)):.3e} "
                      f"max={float(jnp.max(pec_occupancy_local)):.3e}", file=_sys.stderr, flush=True)
                print(f"[kottke debug] inv_xx min={float(jnp.min(_ix)):.3e} "
                      f"max={float(jnp.max(_ix)):.3e} "
                      f"any_nan={bool(jnp.any(jnp.isnan(_ix)))}", file=_sys.stderr, flush=True)
                print(f"[kottke debug] eps_r min={float(jnp.min(materials.eps_r)):.3e} "
                      f"max={float(jnp.max(materials.eps_r)):.3e}", file=_sys.stderr, flush=True)
                print(f"[kottke debug] sigma min={float(jnp.min(materials.sigma)):.3e} "
                      f"max={float(jnp.max(materials.sigma)):.3e}", file=_sys.stderr, flush=True)

        # Build the TRACED RLC metas now that all material folding (RLC + ports)
        # is complete — build_rlc_meta_traced reads the final eps/sigma at each
        # element cell, mirroring the concrete runners/uniform.py:516 ordering.
        if self._lumped_rlc:
            from rfx.lumped import build_rlc_meta_traced
            _rlc_ov = rlc_values_override or {}
            rlc_metas = []
            for _rlc_idx, _rlc_spec in enumerate(self._lumped_rlc):
                _v = _rlc_ov.get(_rlc_idx, {})
                rlc_metas.append(build_rlc_meta_traced(
                    grid, _rlc_spec, materials,
                    r_val=_v.get("R"), l_val=_v.get("L"), c_val=_v.get("C"),
                ))

        result = _run(
            grid,
            materials,
            n_steps,
            boundary=self._boundary,
            cpml_axes=cpml_axes_run,
            pec_axes=pec_axes_run,
            periodic=periodic_bool,
            debye=debye,
            lorentz=lorentz,
            sources=sources,
            probes=probes,
            waveguide_ports=waveguide_ports if waveguide_ports else None,
            ntff=ntff_box,
            checkpoint=checkpoint,
            checkpoint_segments=checkpoint_segments,
            pec_mask=pec_mask_local,
            pec_occupancy=pec_occupancy_for_run,
            aniso_inv_eps=aniso_inv_eps_run,
            aniso_inv_eps_smooth=(aniso_inv_eps_run is not None),
            lumped_port_sparams=lumped_port_sparam_specs or None,
            wire_port_sparams=wire_port_sparam_specs or None,
            wire_refplane_sparams=wire_refplane_specs or None,
            lumped_rlc=rlc_metas,
            dft_planes=dft_planes if dft_planes else None,
            return_state=False,
            stencil_order=self._stencil_order,
        )

        # Multi-drive S-matrix hook (item-5 Stage 1): return the raw all-port
        # (v_dft, i_dft) accumulators for off-line wave decomposition by the
        # production-scan driver, bypassing the diagonal-only extraction below.
        if _return_raw_port_sparams:
            _raw_freqs = None
            if result.lumped_port_sparams:
                _raw_freqs = result.lumped_port_sparams[0][0].freqs
            elif result.wire_port_sparams:
                _raw_freqs = result.wire_port_sparams[0][0].freqs
            return {
                "lumped": result.lumped_port_sparams,
                "wire": result.wire_port_sparams,
                "wire_refplane": result.wire_refplane_sparams,
                "freqs": _raw_freqs,
            }

        s_params_out = getattr(result, "s_params", None)
        freqs_out = getattr(result, "freqs", None)
        if result.lumped_port_sparams:
            from rfx.probes.probes import extract_lumped_s11
            s_list = []
            for spec, accs in result.lumped_port_sparams:
                v_dft, i_dft = accs
                s_list.append(extract_lumped_s11(v_dft, i_dft, z0=spec.impedance))
            s_params_out = s_list[0] if len(s_list) == 1 else jnp.stack(s_list, axis=0)
            freqs_out = result.lumped_port_sparams[0][0].freqs
        elif result.wire_port_sparams:
            # Wire-port wave decomposition uses the same FDTD sign
            # convention as lumped (V = -E·dx). Reuse extract_lumped_s11
            # which implements S11 = (V + Z0·I)/(V − Z0·I).
            from rfx.probes.probes import extract_lumped_s11
            w_list = []
            for spec, accs in result.wire_port_sparams:
                v_dft, i_dft, _v_inc_dft = accs
                w_list.append(extract_lumped_s11(v_dft, i_dft, z0=spec.impedance))
            s_params_out = w_list[0] if len(w_list) == 1 else jnp.stack(w_list, axis=0)
            freqs_out = result.wire_port_sparams[0][0].freqs

        # Passivity self-check (tracer-safe; skipped under jax.grad/jit) — the
        # eager single-cell lumped/wire extractor can return non-physical
        # |S11|>1 where the incident wave is weak (spectral band edges), and
        # forward() previously surfaced no check (unlike compute_*_s_matrix).
        if s_params_out is not None and (
            result.lumped_port_sparams or result.wire_port_sparams
        ):
            from rfx.probes.probes import warn_if_nonpassive_lumped_s11
            warn_if_nonpassive_lumped_s11(
                s_params_out, freqs_out,
                extractor="forward(port_s11_freqs=...)",
            )

        # Convert tuple → name-keyed dict, mirroring runners/uniform.py:704
        # so consumers can index by the same name they registered with.
        dft_planes_out = None
        sim_dft_planes = getattr(result, "dft_planes", None)
        if self._dft_planes and sim_dft_planes:
            dft_planes_out = {
                entry.name: probe
                for entry, probe in zip(self._dft_planes, sim_dft_planes)
            }

        return ForwardResult(
            time_series=result.time_series,
            ntff_data=result.ntff_data,
            ntff_box=result.ntff_box,
            grid=result.grid,
            s_params=s_params_out,
            freqs=freqs_out,
            lumped_port_sparams=result.lumped_port_sparams,
            wire_port_sparams=result.wire_port_sparams,
            dft_planes=dft_planes_out,
        )

    @staticmethod
    def _pack_nu_forward_result(
        *,
        time_series,
        grid,
        ntff_data=None,
        ntff_box=None,
        s_params=None,
        freqs=None,
        dft_planes=None,
    ) -> ForwardResult:
        """Assemble the minimal ``ForwardResult`` for both NU forward lanes.

        The single-device (:meth:`_forward_nonuniform_from_materials`) and
        distributed (:meth:`_forward_distributed_nonuniform_from_materials`)
        lanes are a hand-maintained mirror pair whose only shared concern is
        this final ``ForwardResult`` assembly. Each lane extracts its own
        per-lane field values (the single-device lane reads a ``Result``
        object; the distributed lane reads a runner dict and forces the
        unsupported observables to ``None``) and passes them here as explicit
        parameters — there is no closure capture of caller locals, matching
        the W6.1 ``_StepContext`` / W6.6 builder precedents. Centralising the
        constructor keeps the two lanes' output schema from drifting apart.
        """
        return ForwardResult(
            time_series=time_series,
            ntff_data=ntff_data,
            ntff_box=ntff_box,
            grid=grid,
            s_params=s_params,
            freqs=freqs,
            dft_planes=dft_planes,
        )

    def _forward_nonuniform_from_materials(
        self,
        *,
        eps_override: jnp.ndarray | None = None,
        sigma_override: jnp.ndarray | None = None,
        pec_mask_override: jnp.ndarray | None = None,
        pec_occupancy_override: jnp.ndarray | None = None,
        n_steps: int,
        checkpoint: bool = True,
        emit_time_series: bool = True,
        checkpoint_every: int | None = None,
        n_warmup: int = 0,
        design_mask: jnp.ndarray | None = None,
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

        result = run_nonuniform_path(
            self,
            n_steps=n_steps,
            eps_override=eps_override,
            sigma_override=sigma_override,
            pec_mask_override=pec_mask_override,
            pec_occupancy_override=pec_occupancy_override,
            checkpoint=checkpoint,
            emit_time_series=emit_time_series,
            checkpoint_every=checkpoint_every,
            n_warmup=n_warmup,
            design_mask=design_mask,
        )
        return self._pack_nu_forward_result(
            time_series=result.time_series,
            ntff_data=result.ntff_data,
            ntff_box=result.ntff_box,
            grid=result.grid,
            s_params=getattr(result, "s_params", None),
            freqs=getattr(result, "freqs", None),
            dft_planes=getattr(result, "dft_planes", None),
        )

    def _forward_distributed_nonuniform_from_materials(
        self,
        *,
        eps_override: jnp.ndarray | None = None,
        sigma_override: jnp.ndarray | None = None,
        pec_mask_override: jnp.ndarray | None = None,
        pec_occupancy_override: jnp.ndarray | None = None,
        n_steps: int,
        checkpoint: bool = True,
        emit_time_series: bool = True,
        checkpoint_every: int | None = None,
        n_warmup: int = 0,
        design_mask: jnp.ndarray | None = None,
        devices: list | None = None,
        exchange_interval: int = 1,
        skip_preflight: bool = False,
    ) -> ForwardResult:
        """Phase 3 (issue #44): differentiable forward on the **distributed**
        non-uniform mesh path.

        The single-device sibling :meth:`_forward_nonuniform_from_materials`
        delegates the whole pipeline to ``run_nonuniform_path``; this lane
        instead inlines material assembly + sharding because the sharded
        runner (``rfx.runners.distributed_nu.run_nonuniform_distributed_pec``)
        takes pre-sharded inputs. The two lanes share only the final
        ``ForwardResult`` assembly, factored into
        :meth:`_pack_nu_forward_result`. This lane uses x-axis 1-D slab
        decomposition across ``devices``.  Performs
        the V3 §M4 distributed-specific preflight (5 checks) before any
        trace build, then assembles materials, builds the
        :class:`ShardedNUGrid`, shards every input array, and calls the
        Phase 2F runner.

        See :meth:`forward` for the public-facing kwarg semantics.
        """
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
        import warnings as _w
        from rfx.runners.distributed_nu import (
            build_sharded_nu_grid,
            init_cpml_for_sharded_nu,
            run_nonuniform_distributed_pec,
            shard_cpml_state_x_slab,
            shard_debye_coeffs_x_slab,
            shard_debye_state_x_slab,
            shard_design_mask_x_slab,
            shard_lorentz_coeffs_x_slab,
            shard_lorentz_state_x_slab,
            shard_pec_mask_x_slab,
            shard_pec_occupancy_x_slab,
        )
        from rfx.core.yee import MaterialArrays
        from rfx.nonuniform import (
            position_to_index as _nu_pos_to_idx,
            make_current_source as _nu_make_current_source,
        )
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

            # Check 4 — CPML vs local slab on outer boundary ranks.
            cpml_layers = int(getattr(self, "_cpml_layers", 0) or 0)
            if self._boundary == "cpml" and cpml_layers > 0:
                for rank in (0, n_devices - 1):
                    nx_local_real = nx_per_rank
                    if cpml_layers * 2 >= nx_local_real:
                        raise ValueError(
                            f"cpml_layers*2={cpml_layers * 2} >= "
                            f"nx_local={nx_local_real} on boundary rank "
                            f"{rank}; reduce cpml_layers (or set per-face "
                            f"lo_thickness/hi_thickness on the x Boundary) "
                            f"or increase nx."
                        )

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

        # ---- Assemble full-domain materials ----
        materials, debye_spec, lorentz_spec, pec_mask = (
            self._assemble_materials_nu(grid)
        )

        # ``eps_override`` / ``sigma_override`` may be JAX tracers (the
        # caller is differentiating w.r.t. eps/sigma).  Keep the original
        # concrete materials for ``make_current_source`` so source
        # normalisation stays Python-float (matches the single-device NU
        # runner's ``materials_concrete`` pattern in run_nonuniform_path).
        materials_concrete = materials
        if eps_override is not None or sigma_override is not None:
            materials = MaterialArrays(
                eps_r=(
                    eps_override if eps_override is not None
                    else materials.eps_r
                ),
                sigma=(
                    sigma_override if sigma_override is not None
                    else materials.sigma
                ),
                mu_r=materials.mu_r,
            )
        if pec_mask_override is not None:
            pec_mask = (
                pec_mask_override if pec_mask is None
                else (pec_mask | pec_mask_override)
            )

        # ---- Initialise Debye / Lorentz state on the full domain BEFORE
        # sharding (distributed_nu shard helpers expect the full-domain
        # arrays produced by init_debye / init_lorentz). ----
        debye = None
        if debye_spec is not None:
            debye_poles, debye_masks = debye_spec
            debye = init_debye(
                debye_poles, materials, grid.dt, mask=debye_masks,
            )
        lorentz = None
        if lorentz_spec is not None:
            lorentz_poles, lorentz_masks = lorentz_spec
            lorentz = init_lorentz(
                lorentz_poles, materials, grid.dt, mask=lorentz_masks,
            )

        # ---- Build sharded grid + mesh ----
        sharded_grid = build_sharded_nu_grid(
            grid, n_devices, exchange_interval=exchange_interval,
        )
        from jax.sharding import Mesh
        mesh = Mesh(np.array(devices), axis_names=("x",))

        # ---- Shard materials.  ``_split_materials`` lives in
        # rfx.runners.distributed and pads the high-x end before slabbing. ----
        from rfx.runners.distributed import _split_materials
        from jax.sharding import NamedSharding, PartitionSpec as _P
        shd = NamedSharding(mesh, _P("x"))
        nx = grid.nx
        pad_x = sharded_grid.pad_x
        if pad_x > 0:
            _pad_widths = ((0, pad_x), (0, 0), (0, 0))
            materials_padded = MaterialArrays(
                eps_r=jnp.pad(materials.eps_r, _pad_widths,
                              constant_values=1.0),
                sigma=jnp.pad(materials.sigma, _pad_widths,
                              constant_values=0.0),
                mu_r=jnp.pad(materials.mu_r, _pad_widths,
                             constant_values=1.0),
            )
        else:
            materials_padded = materials

        ghost = sharded_grid.ghost_width
        materials_slabs = _split_materials(
            materials_padded, n_devices, ghost,
        )

        def _shard_3d_stacked(arr):
            n_dev = arr.shape[0]
            rest = arr.shape[1:]
            return jax.device_put(
                arr.reshape(n_dev * rest[0], *rest[1:]), shd,
            )

        sharded_materials = MaterialArrays(
            eps_r=_shard_3d_stacked(materials_slabs.eps_r),
            sigma=_shard_3d_stacked(materials_slabs.sigma),
            mu_r=_shard_3d_stacked(materials_slabs.mu_r),
        )

        # ---- Shard PEC mask / occupancy / design mask via Phase 2 helpers. ----
        sharded_pec_mask = shard_pec_mask_x_slab(pec_mask, sharded_grid)
        sharded_pec_occupancy = shard_pec_occupancy_x_slab(
            pec_occupancy_override, sharded_grid,
        )
        sharded_design_mask = shard_design_mask_x_slab(
            design_mask, sharded_grid,
        )

        # ---- CPML init + sharding (Phase 2C). ----
        cpml_params = None
        cpml_state_sharded = None
        cpml_layers = int(getattr(self, "_cpml_layers", 0) or 0)
        if self._boundary == "cpml" and cpml_layers > 0:
            cpml_params, cpml_state_stacked = init_cpml_for_sharded_nu(
                sharded_grid, n_devices,
                pec_faces=getattr(self, "_pec_faces", None),
            )
            cpml_state_sharded = shard_cpml_state_x_slab(
                cpml_state_stacked, sharded_grid, mesh,
            )

        # ---- Shard Debye / Lorentz dispersion (Phase 2D). ----
        sharded_debye = None
        if debye is not None:
            db_coeffs, db_state = debye
            sharded_debye = (
                shard_debye_coeffs_x_slab(db_coeffs, sharded_grid, mesh),
                shard_debye_state_x_slab(db_state, sharded_grid, mesh),
            )
        sharded_lorentz = None
        if lorentz is not None:
            lr_coeffs, lr_state = lorentz
            sharded_lorentz = (
                shard_lorentz_coeffs_x_slab(lr_coeffs, sharded_grid, mesh),
                shard_lorentz_state_x_slab(lr_state, sharded_grid, mesh),
            )

        # ---- Sources / probes (lumped/wire/coax ports unsupported here). ----
        if self._lumped_rlc:
            raise NotImplementedError(
                "Lumped RLC ports are not yet supported on the "
                "distributed=True forward path; remove the lumped_rlc "
                "spec or omit distributed=True."
            )
        sources: list[SourceSpec] = []
        for pe in self._ports:
            if pe.impedance > 0.0:
                raise NotImplementedError(
                    "Lumped / wire ports (impedance > 0) are not yet "
                    "supported on the distributed=True forward path; "
                    "use distributed=False or replace with a current "
                    "source (impedance=0)."
                )
            idx = _nu_pos_to_idx(grid, pe.position)
            # Use concrete materials so make_current_source can resolve
            # eps / sigma to Python floats (it calls ``float(...)`` on
            # both for the source-cell normalisation).
            si, sj, sk, sc, wf = _nu_make_current_source(
                grid, idx, pe.component, pe.waveform, n_steps,
                materials_concrete,
            )
            sources.append(SourceSpec(
                i=int(si), j=int(sj), k=int(sk),
                component=sc, waveform=jnp.asarray(wf),
            ))

        probes: list[ProbeSpec] = []
        for pe in self._probes:
            idx = _nu_pos_to_idx(grid, pe.position)
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
            sharded_design_mask=sharded_design_mask,
            emit_time_series=emit_time_series,
            pmc_faces=frozenset(self._boundary_spec.pmc_faces()),
        )

        # ---- Repackage into ForwardResult via the shared lane helper.
        # Both the distributed runner and the single-device NU runner
        # return time_series with layout ``(n_steps, n_probes)``; we
        # surface that schema unchanged so vmap_sweep / decay_convergence
        # / lumped_rlc / etc. continue to work. The distributed lane has no
        # NTFF / S-param / freq observables, so those stay ``None``.
        ts = result.get("time_series")
        return self._pack_nu_forward_result(
            time_series=ts,
            grid=grid,
            ntff_data=None,
            ntff_box=None,
            s_params=None,
            freqs=None,
            dft_planes=result.get("dft_planes")
                if hasattr(result, "get") else None,
        )

    # ---- unified lane dispatch (W6.3) ----

    def _nu_n_steps(self, num_periods: float) -> int:
        """Resolve the timestep count for a non-uniform / distributed-NU
        lane from a throwaway NU grid build.

        Pure: ``_build_nonuniform_grid()`` synthesises any missing dz/dx/dy
        profile internally without mutating sim state (W1.3). The single
        formula here replaces the four byte-identical
        ``int(np.ceil(num_periods / (freq_max * grid.dt)))`` snippets that
        used to live at each NU/distributed dispatch site.
        """
        grid = self._build_nonuniform_grid()
        return int(np.ceil(num_periods / (self._freq_max * grid.dt)))

    def _dispatch_plan(
        self,
        *,
        mode: str,
        n_steps: int | None,
        num_periods: float,
        # forward-only inputs
        distributed: bool = False,
        port_s11_freqs: object | None = None,
        checkpoint_segments: int | None = None,
        emit_time_series: bool = True,
        checkpoint_every: int | None = None,
        design_mask: object | None = None,
        # run-only inputs
        devices: list | None = None,
        exchange_interval: int = 1,
    ) -> _DispatchPlan:
        """Select the execution lane and reject unsupported config combos.

        The single decision-and-rejection point consumed by both
        :meth:`run` and :meth:`forward` (W6.3). All lane-rejection guards
        (``NotImplementedError`` for unsupported combinations,
        distributed-lane ``ValueError`` guardrails) live here so there is
        exactly one place that decides a lane and refuses an impossible one.

        ``mode`` is ``"forward"`` or ``"run"``. The two modes share the
        ``is_nonuniform`` boolean and the NU ``n_steps`` formula but have
        disjoint lane-token sets:

        - forward: ``fwd_distributed_nu`` / ``fwd_nonuniform`` / ``fwd_uniform``
        - run: ``run_distributed`` / ``run_nonuniform`` / ``run_adi`` /
          ``run_subgridded`` / ``run_uniform``

        ``n_steps`` is returned resolved for the NU/distributed lanes (whose
        step count comes from a throwaway grid) and ``None`` for lanes that
        build-and-reuse a grid (the caller resolves it there).
        """
        is_nonuniform = (
            self._dz_profile is not None
            or self._dx_profile is not None
            or self._dy_profile is not None
        )

        if mode == "forward":
            def _fwd_nu_n_steps() -> int:
                # Throwaway NU grid build for the step count; mirrors the
                # original forward() inline form (period = 1/freq_max).
                grid_probe = self._build_nonuniform_grid()
                period = 1.0 / float(self._freq_max)
                return int(np.ceil(
                    num_periods * period / float(grid_probe.dt)))

            # Issue #72: forward(port_s11_freqs=...) is currently wired only on
            # the uniform single-device path. Reject loudly elsewhere so users
            # don't get a silent s_params=None.
            if port_s11_freqs is not None and (distributed or is_nonuniform):
                raise NotImplementedError(
                    "forward(port_s11_freqs=...) is currently wired only on the "
                    "uniform single-device forward path (issue #72). Drop "
                    "port_s11_freqs or run on a uniform mesh without "
                    "distributed=True."
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
                _n = n_steps if n_steps is not None else _fwd_nu_n_steps()
                return _DispatchPlan(lane="fwd_distributed_nu", n_steps=_n)

            if is_nonuniform:
                # Let the NU runner build grid/materials so it can apply the
                # NU-aware pec_mask and port/source setup against per-axis widths.
                _n = n_steps if n_steps is not None else _fwd_nu_n_steps()
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
            if design_mask is not None:
                raise NotImplementedError(
                    "design_mask (issue #41) is currently only supported on the "
                    "non-uniform forward path. Ping #41 if you need it on the "
                    "uniform path — the same step_fn stop_gradient pattern applies."
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

        # ---- Distributed + nonuniform (Phase B guardrail).
        # Phase B permits the combination for PEC boundary with grading
        # ratio <= 5 and no TFSF. The distributed_v2 runner dispatches to
        # the NU kernels in distributed_nu.py; dispersion and CPML on the
        # distributed NU path are Phase C items and still raise below.
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
            return _DispatchPlan(lane="run_distributed", n_steps=_n)

        # ---- Non-uniform mesh lane ----
        if is_nonuniform:
            _n = n_steps
            if _n is None:
                _n = self._nu_n_steps(num_periods)
            return _DispatchPlan(lane="run_nonuniform", n_steps=_n)

        # ---- ADI lane (n_steps resolved by caller from the reused grid) ----
        if self._solver == "adi":
            return _DispatchPlan(lane="run_adi", n_steps=n_steps)

        # ---- Subgridded lane (n_steps resolved by caller — refinement ratio
        # scaling needs the reused grid) ----
        if self._refinement is not None:
            return _DispatchPlan(lane="run_subgridded", n_steps=n_steps)

        # ---- Uniform lane (n_steps resolved by caller from the reused grid) ----
        return _DispatchPlan(lane="run_uniform", n_steps=n_steps)

    # ---- forward (differentiable) ----

    def forward(
        self,
        *,
        eps_override: jnp.ndarray | None = None,
        sigma_override: jnp.ndarray | None = None,
        pec_mask_override: jnp.ndarray | None = None,
        pec_occupancy_override: jnp.ndarray | None = None,
        n_steps: int | None = None,
        num_periods: float = 20.0,
        checkpoint: bool = True,
        checkpoint_segments: int | None = None,
        emit_time_series: bool = True,
        checkpoint_every: int | None = None,
        n_warmup: int = 0,
        skip_preflight: bool = False,
        design_mask: jnp.ndarray | None = None,
        distributed: bool = False,
        devices: list | None = None,
        exchange_interval: int = 1,
        port_s11_freqs: object | None = None,
        rlc_values_override: dict | None = None,
    ) -> ForwardResult:
        """Run a minimal differentiable forward simulation.

        This path is designed for ``jax.grad`` / ``jax.value_and_grad`` and
        intentionally returns only the observables needed by differentiable
        objectives instead of the broader stateful :meth:`run` result.

        Parameters
        ----------
        eps_override : jnp.ndarray or None
            Replacement permittivity array with shape ``grid.shape``.
        sigma_override : jnp.ndarray or None
            Replacement conductivity array with shape ``grid.shape``.
        pec_mask_override : jnp.ndarray or None
            Additional hard PEC mask to merge with geometry-defined PEC.
        pec_occupancy_override : jnp.ndarray or None
            Relaxed conductor occupancy field in ``[0, 1]`` for
            differentiable PEC-style optimisation.
        n_steps : int or None
            Number of timesteps. If None, auto-computed from *num_periods*.
        num_periods : float
            Number of periods at freq_max for auto step count.
        checkpoint : bool
            Enable gradient checkpointing (default True).
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
        skip_preflight : bool
            Skip the consolidated :meth:`preflight` validation suite that
            normally runs before the forward simulation (default False).
            Use only when preflight has already been run by the caller
            (e.g. :func:`rfx.optimize` runs it once at entry) or to bypass a
            known-spurious warning on a deliberate configuration.
        design_mask : jnp.ndarray or None
            Boolean array with shape ``grid.shape`` selecting the
            differentiable design region (issue #41).  Cells where the mask
            is ``True`` keep their gradient; cells where it is ``False`` have
            ``stop_gradient`` applied each step, so AD memory tracks only the
            design subvolume while forward physics is bit-identical.
            Currently only supported on the non-uniform forward path; on the
            uniform path it raises ``NotImplementedError``.
        distributed : bool, optional
            **Opt-in, unstable, pending GPU evidence (issue #44).**  When
            ``True`` and a non-uniform mesh is configured, route the
            differentiable forward through the sharded NU runner
            (``rfx.runners.distributed_nu.run_nonuniform_distributed_pec``)
            using a 1-D x-slab decomposition across ``devices``.  Defaults
            to ``False`` (single-device path, no behaviour change).  In
            v1.6.2 the distributed forward path is **NU-only** (DP3): a
            uniform mesh raises ``NotImplementedError``.  TFSF sources
            and waveguide ports are unsupported on this path and raise
            ``NotImplementedError`` at preflight.
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
            wired only on the uniform single-device path; non-uniform meshes
            and ``distributed=True`` raise ``NotImplementedError``.
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

        Returns
        -------
        ForwardResult
            Minimal differentiable observables (time series and optional NTFF).
        """
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
                "add_coaxial_port() is not wired into Simulation.forward() "
                "as a validated high-level source/port path. Use "
                "add_port(..., extent=...) for differentiable probe-feed "
                "S11 objectives."
            )

        if port_s11_freqs is not None:
            self._validate_forward_sparameter_request()

        self._auto_preflight(skip=skip_preflight, context="forward")

        # ---- (2,4) stencil fence: reject order=4 on unsupported lanes ----
        self._check_stencil_order_supported(distributed=distributed)

        # ---- W6.3 unified lane dispatch: one place decides + rejects ----
        plan = self._dispatch_plan(
            mode="forward",
            n_steps=n_steps,
            num_periods=num_periods,
            distributed=distributed,
            port_s11_freqs=port_s11_freqs,
            checkpoint_segments=checkpoint_segments,
            emit_time_series=emit_time_series,
            checkpoint_every=checkpoint_every,
            design_mask=design_mask,
        )

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

        if plan.lane == "fwd_distributed_nu":
            return self._forward_distributed_nonuniform_from_materials(
                eps_override=eps_override,
                sigma_override=sigma_override,
                pec_mask_override=pec_mask_override,
                pec_occupancy_override=pec_occupancy_override,
                n_steps=plan.n_steps,
                checkpoint=checkpoint,
                emit_time_series=emit_time_series,
                checkpoint_every=checkpoint_every,
                n_warmup=n_warmup,
                design_mask=design_mask,
                devices=devices,
                exchange_interval=exchange_interval,
                skip_preflight=skip_preflight,
            )

        if plan.lane == "fwd_nonuniform":
            return self._forward_nonuniform_from_materials(
                eps_override=eps_override,
                sigma_override=sigma_override,
                pec_mask_override=pec_mask_override,
                pec_occupancy_override=pec_occupancy_override,
                n_steps=plan.n_steps,
                checkpoint=checkpoint,
                emit_time_series=emit_time_series,
                checkpoint_every=checkpoint_every,
                n_warmup=n_warmup,
                design_mask=design_mask,
            )

        # ---- Uniform forward lane (plan.lane == "fwd_uniform") ----
        n_steps = plan.n_steps
        grid = self._build_grid()
        materials, debye_spec, lorentz_spec, pec_mask, _, _, _ = self._assemble_materials(grid)

        if eps_override is not None or sigma_override is not None:
            materials = MaterialArrays(
                eps_r=eps_override if eps_override is not None else materials.eps_r,
                sigma=sigma_override if sigma_override is not None else materials.sigma,
                mu_r=materials.mu_r,
            )

        if pec_mask_override is not None:
            pec_mask = pec_mask_override if pec_mask is None else (pec_mask | pec_mask_override)

        if n_steps is None:
            n_steps = grid.num_timesteps(num_periods=num_periods)

        _res = self._forward_from_materials(
            grid,
            materials,
            debye_spec,
            lorentz_spec,
            n_steps=n_steps,
            checkpoint=checkpoint,
            checkpoint_segments=checkpoint_segments,
            pec_mask=pec_mask,
            pec_occupancy=pec_occupancy_override,
            port_s11_freqs=port_s11_freqs,
            rlc_values_override=rlc_values_override,
        )
        _warn_if_nonfinite_result(_res, context="forward")
        return _res

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
            When provided, overrides *n_steps* and runs until field
            energy decays to this fraction of peak. E.g. ``1e-3``.
            **Boundary-dependent stop (issue #169, RESOLVED for absorbing
            boundaries):** On **absorbing** boundaries (``cpml`` / ``upml``)
            the stop now uses the **total interior-domain energy**
            ``U = sum(E^2 + H^2)`` over the non-CPML interior — a whole-domain
            quantity that does not pass through the per-cell interference nulls
            of the old single-cell point-field stop, so it **is** suitable for
            flux / S-parameter / transmission measurements on guided / low-loss
            geometries (#169 resolved). On **closed / PEC** boundaries domain
            energy does not decay, so the stop falls back to the historical
            *instantaneous* single-cell field; that fallback keeps the original
            limitation (valid only for lossy / radiating ring-down, not flux /
            S-param / transmission gating on guided / low-loss closed geometries
            — use a fixed ``n_steps`` there, see
            ``validation/crossval/03_straight_waveguide_flux.py`` and the
            :func:`rfx.simulation.run_until_decay` note).
            **Non-uniform (dx/dy/dz-profile) meshes (issue #383):** the same
            interior-energy stop runs on absorbing (``cpml``/``upml``)
            boundaries via a chunked host loop over the NU scan step
            (dV-weighted energy — per-cell ``dx*dy*dz`` — so graded cells
            are weighted faithfully). Closed-boundary NU sims warn and run
            the fixed ``n_steps`` (no point-field fallback on the NU lane).
            On the NU decay path all step-sized buffers are allocated at
            ``decay_max_steps`` and flux monitors must keep the default
            rectangular DFT window. (The non-uniform runner additionally
            rejects its internal ``checkpoint_every`` segmentation option
            when a decay stop is requested; that option belongs to the
            ``forward()`` lane and is not reachable through ``run()``.)
            **DC floor (issue #388):** a soft-source waveform with net DC
            content deposits a static end-charge whose self-energy floors
            the interior-energy criterion; ``run()`` warns quantitatively
            when a planned soft-source waveform's relative DC exceeds 1e-3
            (raise the ``GaussianPulse`` cutoff, narrow a
            ``ModulatedGaussian`` to bandwidth <= 0.4, or use a fixed
            ``n_steps``). This advisory does NOT catch low-rel_DC floors
            amplified by thin source cells (the #388 patch class measured
            the floor at rel_DC ~6e-5) — see issue #388.
        decay_check_interval : int
            Check decay every N steps (default 50).
        decay_min_steps : int
            Always run at least this many steps (default 100).
        decay_max_steps : int
            Hard upper limit on steps (default 50000).
        decay_energy_consecutive : int
            Absorbing-boundary only: number of consecutive sub-threshold
            interior-energy checks required before stopping (default ``2``;
            ``>= 2`` mandatory — the interior energy is not null-free and a
            single check can false-fire on a transient inter-packet dip).
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
            along the x-axis (via ``jax.pmap``).  Phase 1 supports PEC
            boundary, soft sources, and point probes.
        exchange_interval : int, optional
            How often (in timesteps) to perform ghost cell exchange in
            the distributed runner.  Default 1 (every step).  Higher
            values (2-4) reduce synchronization overhead at the cost of
            O(interval * dt) boundary error.

        Returns
        -------
        Result
        """
        fixed_num_periods = n_steps is None

        # Behaviour-neutral decay-parameter sanity advisories (post-#392
        # review): single run()-level site, before dispatch, so both lanes
        # that honour until_decay (uniform + absorbing-boundary
        # non-uniform) are covered by one check.
        self._warn_decay_param_sanity(
            until_decay=until_decay,
            decay_min_steps=decay_min_steps,
            decay_max_steps=decay_max_steps,
        )

        # ---- P1: Auto mesh when dx not specified and geometry exists ----
        if self._dx is None and (self._geometry or self._thin_conductors):
            self._auto_configure_mesh()

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
                "add_coaxial_port() is not wired into Simulation.run() as a "
                "validated high-level source/port path. Use "
                "add_port(..., extent=...) for current claims-bearing "
                "probe-feed S-parameters, or the low-level "
                "rfx.sources.coaxial_port helpers for diagnostic material/"
                "source experiments."
            )

        self._validate_run_sparameter_request(
            compute_s_params=compute_s_params,
            s_param_freqs=s_param_freqs,
            s_param_n_steps=s_param_n_steps,
            devices=devices,
        )

        # ---- P0: Pre-simulation validation ----
        # Run the SAME consolidated, skippable preflight that forward() gets
        # (issue #66 parity): _auto_preflight wraps preflight() — mesh quality,
        # simulation config AND the NTFF/inverse-design check — into one
        # UserWarning, and is robust under tracing (it try/except-wraps
        # preflight). Previously run() called only the mesh + config validators
        # directly, as scattered raw warnings with no skip_preflight control,
        # so the documented lumped/wire S-parameter path via
        # run(compute_s_params=True) silently missed part of the best
        # proactive error surface in the codebase.
        # check_ntff=False: run() historically never ran the inverse-design
        # NTFF PEC-overlap check; keep that surface (it belongs to
        # forward(port_s11_freqs=...)/optimize). Avoids hard-failing run() on
        # an NTFF-box-crosses-PEC config that completed before this change.
        self._auto_preflight(skip=skip_preflight, context="run", check_ntff="advisory")

        # ---- (2,4) stencil fence: reject order=4 on unsupported lanes ----
        _distributed_run = devices is not None and len(devices) > 1
        self._check_stencil_order_supported(distributed=_distributed_run)

        # ---- W6.3 unified lane dispatch: one place decides + rejects ----
        # _dispatch_plan computes the is_nonuniform/_nu_profile boolean, runs
        # the distributed adi/upml ValueErrors + the distributed+NU grading
        # guardrail, selects the lane, and resolves n_steps for the
        # NU/distributed lanes (the uniform/adi/subgridded lanes resolve it
        # below from the grid they build and reuse).
        plan = self._dispatch_plan(
            mode="run",
            n_steps=n_steps,
            num_periods=num_periods,
            devices=devices,
            exchange_interval=exchange_interval,
        )
        n_steps = plan.n_steps

        # ---- Distributed multi-device lane ----
        if plan.lane == "run_distributed":
            self._warn_unsupported_run_kwargs("distributed multi-device", {
                "subpixel_smoothing": subpixel_smoothing,
                "checkpoint": checkpoint,
                "snapshot": snapshot,
                "until_decay": until_decay,
                "conformal_pec": conformal_pec,
                "compute_s_params": compute_s_params,
                "s_param_freqs": s_param_freqs,
                "s_param_n_steps": s_param_n_steps,
            })
            from rfx.runners.distributed_v2 import run_distributed
            _res = run_distributed(
                self, n_steps=n_steps, devices=devices,
                exchange_interval=exchange_interval,
            )
            _warn_if_nonfinite_result(_res, context="run")
            return _res

        # ---- Non-uniform mesh lane ----
        if plan.lane == "run_nonuniform":
            # #383: until_decay is supported on the NU lane for absorbing
            # boundaries (the interior-energy criterion, same class as the
            # uniform #169 stop). Closed/PEC NU domains keep warn-and-drop
            # with a lane-accurate reason: their interior energy does not
            # decay, and the NU lane has no point-field fallback.
            _nu_until_decay = (
                until_decay
                if self._boundary in ("cpml", "upml")
                else None
            )
            _nu_dropped = {
                "snapshot": snapshot,
                "conformal_pec": conformal_pec,
            }
            if _nu_until_decay is None:
                _nu_dropped["until_decay"] = until_decay
            self._warn_unsupported_run_kwargs(
                "non-uniform mesh", _nu_dropped,
                reason_overrides={
                    "until_decay":
                        "the interior-energy decay stop needs absorbing "
                        "boundaries (cpml/upml); this closed-boundary "
                        "non-uniform run executes a fixed n_steps, and the "
                        "non-uniform lane has no point-field fallback "
                        "(issue #383)",
                },
            )
            if _nu_until_decay is not None:
                # #388 DC-floor predictor for the NU decay stop. The grid
                # build is pure (no sim-state mutation) and cheap; an
                # advisory must never block the run, so any build failure
                # simply skips the check (the runner will surface it).
                try:
                    _nu_dt_for_dc = self._build_nonuniform_grid().dt
                except Exception:
                    _nu_dt_for_dc = None
                if _nu_dt_for_dc is not None:
                    self._warn_until_decay_dc_floor(
                        dt=_nu_dt_for_dc, n_table=decay_max_steps
                    )
            _res = self._run_nonuniform(
                n_steps=n_steps,
                compute_s_params=compute_s_params,
                s_param_freqs=s_param_freqs,
                subpixel_smoothing=subpixel_smoothing,
                checkpoint=checkpoint,
                until_decay=_nu_until_decay,
                decay_check_interval=decay_check_interval,
                decay_min_steps=decay_min_steps,
                decay_max_steps=decay_max_steps,
                decay_energy_consecutive=decay_energy_consecutive,
            )
            self._warn_run_sparams_if_nonpassive(_res)
            self._warn_postrun_energy_witness(
                _res,
                fixed_num_periods=fixed_num_periods,
                until_decay=until_decay,
            )
            _warn_if_nonfinite_result(_res, context="run")
            return _res

        grid = self._build_grid()
        base_materials, debye_spec, lorentz_spec, pec_mask, pec_shapes, _, kerr_chi3 = self._assemble_materials(grid)

        if plan.lane == "run_adi":
            if until_decay is not None:
                raise ValueError("solver='adi' does not support until_decay yet")
            if snapshot is not None:
                raise ValueError("solver='adi' does not support snapshots yet")
            if n_steps is None:
                n_steps = grid.num_timesteps(num_periods=num_periods)
            _res = self._run_adi_from_materials(
                grid,
                base_materials,
                debye_spec,
                lorentz_spec,
                n_steps=n_steps,
                pec_mask=pec_mask,
                return_state=True,
            )
            _warn_if_nonfinite_result(_res, context="run")
            return _res

        # ---- Subgridded lane ----
        if plan.lane == "run_subgridded":
            self._warn_unsupported_run_kwargs("subgridded (SBP-SAT)", {
                "subpixel_smoothing": subpixel_smoothing,
                "checkpoint": checkpoint,
                "snapshot": snapshot,
                "until_decay": until_decay,
                "conformal_pec": conformal_pec,
            })
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
            _res = self._run_subgridded(
                grid, base_materials, pec_mask,
                n_steps=subgrid_n_steps,
                compute_s_params=compute_s_params,
                s_param_freqs=s_param_freqs,
                s_param_n_steps=s_param_n_steps,
            )
            _warn_if_nonfinite_result(_res, context="run")
            return _res

        # ---- Uniform path ----
        if n_steps is None:
            n_steps = grid.num_timesteps(num_periods=num_periods)

        if until_decay is not None:
            # #388 DC-floor predictor for the decay stop.
            self._warn_until_decay_dc_floor(
                dt=grid.dt, n_table=decay_max_steps
            )

        from rfx.runners.uniform import run_uniform
        _field_dtype = jnp.float16 if self._precision == "mixed" else None
        _res = run_uniform(
            self,
            n_steps=n_steps,
            until_decay=until_decay,
            decay_check_interval=decay_check_interval,
            decay_min_steps=decay_min_steps,
            decay_max_steps=decay_max_steps,
            decay_energy_consecutive=decay_energy_consecutive,
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
            kerr_chi3=kerr_chi3,
            field_dtype=_field_dtype,
        )
        self._warn_run_sparams_if_nonpassive(_res)
        self._warn_postrun_energy_witness(
            _res,
            fixed_num_periods=fixed_num_periods,
            until_decay=until_decay,
        )
        _warn_if_nonfinite_result(_res, context="run")
        return _res
