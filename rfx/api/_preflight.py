"""Preflight and configuration-validation methods for :class:`Simulation`.

Import contract (Part B Stage 1a refactor):
  This module is a transitional mixin. It must import ONLY from
  ``rfx.api._spec`` plus external ``rfx.*`` / stdlib / jax / numpy.
  It must NEVER do ``from rfx.api import ...`` or ``from . import ...``
  the package, to keep ``rfx/api/__init__.py`` the sole composition point.

``Simulation`` inherits ``_PreflightMixin``, so every method below is
bound on ``Simulation`` instances.

This file is a facade and holds three things:

1. The EXECUTION family, which stays here by design and is everything that
   decides WHAT to run rather than checking a configuration: ``preflight``
   and ``preflight_sparameters`` (the entry points), the run / forward
   S-parameter request validators and the three per-calculator
   ``_validate_*_sparameter_request_for_preflight`` routers,
   ``_validate_simulation_config`` — which builds the shared context and
   then iterates ``rfx/preflight/_registry.py``'s ``CORE_CONFIG_CHECKS``,
   whose ORDER is the observable
   ``tests/locks/test_preflight_split_snapshot.py`` pins —
   ``_collect_flux_regions``, and the x64 / settling-witness
   configuration checks.
2. The module-level RE-EXPORT blocks. Every name any leg moved is re-bound
   here, so the module namespace is exactly as wide as it was before the
   split (55 names, pinned by set equality) and the existing readers that do
   ``from rfx.api._preflight import <name>`` keep resolving. Classes are
   re-exported, never redefined: one ``PreflightWarning``, one
   ``_RealizedPEC``.
3. The class-scoped REBINDS. Each moved check body is a module-level ``def``
   in its leg module whose first parameter is still ``self``, imported back
   into the ``_PreflightMixin`` class body AT THE POSITION it held, with
   ``__qualname__`` restored at the leg module's foot and any
   ``@staticmethod`` re-applied here.

Every check family, the conductor-realization layer they all read, and the
check REGISTRY that composes them live in ``rfx/preflight/`` — see that
package's docstring for the nine modules. A new configuration check is added
there and registered; it needs no edit to this file.
"""

from __future__ import annotations

import json
import math
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np

from rfx.grid import C0
from rfx.core.yee import MaterialArrays
from rfx.core.jax_utils import is_tracer
from rfx.geometry.csg import Box


from rfx.preflight._common import (
    _fmt_len,
    _fmt_signed,
    _fmt_freq,
    _absorber_boundary_for_axis,
    _axis_pad_thickness_m,
    _coord_in_absorber,
    _ABSORBER_PROXIMITY_CELLS,
    _coord_near_absorber,
    _sorted_box_corners,
    PreflightWarning,
    PreflightErrorWarning,
    PreflightConfigError,
    PreflightIssue,
    PreflightReport,
    _H_LOOP,
    _component_is_dead,
)


from rfx.preflight.msl import (
    _MSL_REALIZED_THICKNESS_Z0_SENSITIVITY,
    _MSL_REALIZED_THICKNESS_Z0_BUDGET,
    _MSL_REALIZED_THICKNESS_TOL,
    MSL_EPS_EFF_PROXY,
    msl_min_probe_clearance,
    _MSL_NEAR_FIELD_STANDOFF_H_SUB,
    _MSL_NEAR_FIELD_MIN_OFFSET_CELLS,
    msl_source_near_field_standoff_cells,
    msl_nearest_downstream_reflector,
    msl_probe_clearance_for_port,
    msl_absorber_compliant_offset_max,
)


from rfx.preflight.pec_geometry import (
    _CONGRUENCE_EXTENT_QUANTUM_M,
    _CONGRUENCE_SPREAD_TOL_EDGES,
    _CAVITY_THICKNESS_TOL,
    _OFF_LATTICE_EDGE_TOL,
    _CAMPAIGN_MAX_OFFENDERS,
)


from rfx.preflight.waveguide import (
    _waveguide_skipped_note,
    WAVEGUIDE_DEFAULT_NUM_PERIODS,
    resolve_waveguide_port_freqs,
)

from rfx.preflight.mesh import _local_cell


from rfx.preflight.realization import (
    _shape_bounds,
    _shift_back_np,
    _wall_nodes_on_plane,
    _realized_edges_np,
    _RealizedPEC,
    _EntryRealization,
    _CampaignStaticsContext,
)


class _PreflightMixin:
    """Preflight / validation methods mixed into :class:`Simulation`."""

    from rfx.preflight.sources import _validate_tfsf_vacuum_boundary
    _validate_tfsf_vacuum_boundary = staticmethod(_validate_tfsf_vacuum_boundary)

    def _validate_run_sparameter_request(
        self,
        *,
        compute_s_params: bool | None,
        s_param_freqs,
        s_param_n_steps: int | None,
        devices: list | None = None,
    ) -> bool | None:
        """Resolve contaminated default S requests and reject unsupported explicit ones."""

        requested = (
            compute_s_params is True
            or s_param_freqs is not None
            or s_param_n_steps is not None
        )
        from rfx.runners._admission import default_run_forms_s, resolve_plain_source_s_request
        compute_s_params = resolve_plain_source_s_request(
            self, compute_s_params, requested=requested, devices=devices)
        if not (requested or (compute_s_params is None and default_run_forms_s(self, devices=devices))):
            return compute_s_params

        from rfx.runners._admission import admit_s_request
        from rfx.model.coax_admission import refuse_off_record_steps
        refuse_off_record_steps(self, s_param_n_steps)
        port_entries = admit_s_request(self, "run(compute_s_params=True)")

        if devices is not None and len(devices) > 1:
            from rfx.model.distributed_ports import refuse_unsupported_distributed_features
            refuse_unsupported_distributed_features(
                self, lane="distributed multi-device run()")
        if self._refinement is not None and not self._coaxial_ports:
            if any(pe.waveform is None for pe in port_entries):
                raise ValueError(
                    "subgrid compute_s_params needs a waveform "
                    "on every impedance port so each port can be driven in "
                    "turn. Pass waveform=... even for ports whose main-run "
                    "excite flag is False."
                )

        if self._periodic_axes and port_entries:
            raise NotImplementedError(
                "run(compute_s_params=True) for lumped/wire add_port(...) does "
                "not honor periodic axes: the S-parameter extraction re-run uses "
                "non-periodic boundaries, so the returned S-matrix would silently "
                f"ignore BoundarySpec periodic axes {self._periodic_axes!r}. Remove the "
                "periodic axes for the S-parameter run, or use a port family that "
                "supports periodicity (e.g. a Floquet port)."
            )
        return compute_s_params

    def _validate_forward_sparameter_request(self) -> None:
        """Reject ``forward(port_s11_freqs=...)`` outside its narrow path."""

        from rfx.runners._admission import admit_forward_s_matrix, admit_s_request
        admit_forward_s_matrix(self)
        messages: list[str] = []
        if self._msl_ports:
            for pe in self._msl_ports:
                from rfx.preflight._common import (
                    profile_cell_at as _profile_cell_at,
                    profile_node_at as _profile_node_at,
                    profile_span_is_uniform as _profile_span_is_uniform,
                )
                from rfx.preflight.msl import (
                    msl_axis_runs_interval_solve as _runs_interval_solve,
                    msl_auto_probe_ladder as _auto_ladder,
                    msl_auto_probe_offset_cells as _auto_offset_cells,
                    msl_auto_probe_offset_term as _auto_offset_term,
                )
                from rfx.sources.msl_port import (
                    msl_axis_roles as _msl_axis_roles,
                )
                _prop_axis, _, _, _prop_sign = _msl_axis_roles(pe.direction)
                _ax_i = "xyz".index(_prop_axis)
                _runway_profile = (self._dx_profile, self._dy_profile,
                                   self._dz_profile)[_ax_i]
                _feed_coord = float(pe.position[_ax_i])
                _feed_node = _profile_node_at(
                    self._dx or 0.0, _runway_profile, _feed_coord)
                _runway_dx = _profile_cell_at(
                    self._dx or 0.0, _runway_profile, _feed_node,
                    toward="hi" if _prop_sign > 0 else "lo")
                _nf_cells = msl_source_near_field_standoff_cells(
                    float(pe.height), _runway_dx)
                _standoff_len = _nf_cells * _runway_dx * int(_prop_sign)
                if not _profile_span_is_uniform(
                        self._dx or 0.0, _runway_profile,
                        _feed_node, _standoff_len):
                    messages.append(
                        f"MSL port {pe.name!r}: the source-fringing standoff "
                        f"({abs(_standoff_len)*1e3:.3g} mm from the feed plane) "
                        f"crosses cells of more than one size on the "
                        f"{_prop_axis} runway, so a probe-offset in CELLS "
                        "does not name one distance. Put the port and its "
                        "probes inside one uniform zone of the profile."
                    )
                elif self._dx and pe.n_probe_offset is not None:
                    _auto = pe.name in getattr(self, "_msl_auto_offset_min", {})
                    _lengths = getattr(
                        self, "_msl_auto_probe_lengths", {}).get(pe.name)
                    _off = int(pe.n_probe_offset)
                    _none_txt = ""
                    _none_clears = False
                    if _lengths is not None:
                        _sp_auto = pe.name in getattr(
                            self, "_msl_auto_probe_spacing", {})
                        _none_off, _, _none_cell, _none_on_runway = (
                            _auto_ladder(
                                _runway_profile, float(self._dx), _feed_coord,
                                _prop_sign, int(pe.n_probes), _lengths[0],
                                float(pe.height), _lengths[1],
                                n_probe_spacing=(None if _sp_auto
                                                 else pe.n_probe_spacing)))
                        if not _none_on_runway:
                            _none_cell = float(self._dx)
                            _none_off = _auto_offset_cells(
                                _lengths[0], float(pe.height), _none_cell)
                        if _auto:
                            _off = _none_off
                        _none_clears = _none_off >= _nf_cells
                        _none_term = _auto_offset_term(
                            _lengths[0], float(pe.height), _none_cell)
                        _none_txt = (
                            f"counts {_none_term} "
                            + (f"in this runway's {_fmt_len(_none_cell)} "
                               f"cells" if _none_on_runway else
                               "in the scalar dx cell because counted in this "
                               "runway's own cells its probe ladder would "
                               "cross a grading ramp")
                            + (", at least " if _none_on_runway
                               and _runs_interval_solve(_runway_profile)
                               else ", ")
                            + f"{_none_off} cells")
                    if _off < _nf_cells:
                        _remedy = (
                            f"set n_probe_offset >= {_nf_cells} explicitly; the "
                            f"automatic choice {_none_txt}, and leaving it None "
                            f"chooses {_off} again"
                            if _auto else
                            f"increase n_probe_offset or leave it None: the "
                            f"automatic offset {_none_txt}"
                            if _none_clears else
                            f"set n_probe_offset >= {_nf_cells}; leaving it "
                            f"None {_none_txt}, and falls short on this runway"
                            if _none_txt else
                            f"set n_probe_offset >= {_nf_cells}"
                        )
                        messages.append(
                            f"MSL port {pe.name!r}: n_probe_offset="
                            f"{_off} sits within the source fringing "
                            f"transient ({_nf_cells} cells = max(3, round("
                            f"5·h_sub/runway_cell))); probe 0 may corrupt the V·I-split "
                            f"S11 of a high-Q resonant load — "
                            + _remedy
                        )
        admit_s_request(self, "forward(port_s11_freqs=...)", details=messages)

    from rfx.preflight.mesh import (
        _validate_mesh_quality,
        _check_numerical_dispersion,
        _validate_thin_metal_on_nu_mesh,
    )

    from rfx.preflight.realization import (
        _assemble_realized,
        _port_realized_edges,
        _port_pec_mask,
    )
    from rfx.preflight.waveguide import (
        _port_transverse_spans,
        _check_waveguide_port_aperture_snap,
    )


    from rfx.preflight.waveguide import (
        _emit_waveguide_port_cutoff_findings,
        _check_waveguide_port_evanescent_declared_geometry,
        _check_waveguide_port_evanescent,
    )

    from rfx.preflight._flux_subjects import _collect_flux_regions  # type: ignore[misc]

    def preflight(
        self,
        *,
        strict: bool = False,
        check_ntff: bool | str = True,
        check_resolution: bool = True,
        check_ad_memory: bool = False,
        n_steps_for_memory: int | None = None,
        available_memory_gb: float | None = None,
    ) -> "PreflightReport":
        """Run all pre-simulation checks and return warnings.

        Resolves and caches the mesh from the current static declaration,
        without changing declared dx/domain/profiles. Auto-mesh selection is
        emitted as a UserWarning, separately from validation findings.

        Parameters
        ----------
        strict : bool
            If True, collect every finding and raise one ValueError
            that lists them all.
        check_ntff : bool or "advisory"
            ``True`` (default): run the full NTFF check family (PEC-overlap
            hard error + λ/4 / λ/2 near-field gap advisories + the
            sub-wavelength ground-plane pattern advisory).
            ``"advisory"``: run only the advisories — the tier ``run()``
            uses, because the λ/4 and small-ground-plane warnings are
            physics-relevant to any far-field computation while the
            PEC-overlap hard error remains an inverse-design gate.
            ``False``: skip the family entirely.
        check_resolution : bool
            Run the tightened resolution check (existing _validate_mesh_quality
            uses per-material thresholds already — this flag kept for
            symmetry and future tightening). Default True.
        check_ad_memory : bool
            Run AD memory estimate and warn if > 85% of available VRAM.
            Requires n_steps_for_memory. Default False (diagnostic only).
        n_steps_for_memory : int or None
            Step count for AD memory sizing. Required when check_ad_memory.
        available_memory_gb : float or None
            Override VRAM detection. If None, best-effort via JAX devices.

        Returns
        -------
        PreflightReport
            A ``list`` subclass of :class:`PreflightIssue` (each a ``str``
            subclass), back-compatible with the legacy ``list[str]`` return.
            Empty if no issues found. Finite flux-window geometry is recorded
            separately in ``flux_regions``, including for issue-free reports.
        """
        return self._preflight_impl(
            strict=strict, check_ntff=check_ntff, check_resolution=check_resolution,
            check_ad_memory=check_ad_memory, n_steps_for_memory=n_steps_for_memory,
            available_memory_gb=available_memory_gb,
        )

    from rfx.preflight._impl import _preflight_impl

    def preflight_sparameters(
        self,
        *,
        calculator: str = "run",
        strict: bool = False,
        normalize: bool | str | None = None,
        include_general: bool = False,
    ) -> "PreflightReport":
        """Preflight the selected S-parameter calculator without running FDTD.

        This is a routing/contract check for the port-family-specific
        S-parameter APIs.  It answers "which calculator should this simulation
        use?" before an expensive run starts:

        - ``calculator="run"`` checks ``run(compute_s_params=True)`` for
          lumped/wire ``add_port(...)`` families.
        - ``calculator="forward"`` checks ``forward(port_s11_freqs=...)`` for
          uniform single-device port S-matrices.
        - ``calculator="msl"`` checks ``compute_msl_s_matrix(...)``.
        - ``calculator="waveguide"`` checks ``compute_waveguide_s_matrix(...)``.

        Parameters
        ----------
        calculator:
            One of ``"run"``, ``"forward"``, ``"msl"``, or ``"waveguide"``
            (the corresponding method names are accepted as aliases).
        strict:
            If True, escalate ERROR-severity findings to a raise: collect
            everything, then raise a single ``ValueError`` listing the errors
            (aggregate-then-raise). The underlying ``NotImplementedError`` is
            recorded as an error-severity issue and re-surfaced as part of that
            aggregated ``ValueError`` (its exact type is not preserved).

            Errors only, not "any issue": the waveguide setup audits report
            advisory and informational findings on a perfectly valid routing,
            and a healthy two-port guide ALWAYS carries the informational
            E-plane note, so escalating on emptiness would make
            ``strict=True`` raise on every correct waveguide setup. Advisories
            are still returned in the report and printed; read them there.
            (This differs from ``preflight(strict=True)``, whose report has no
            informational tier.)
        normalize:
            Waveguide non-uniform preflight uses this to mirror
            ``compute_waveguide_s_matrix(normalize=...)``.  ``None`` means the
            method default, currently ``False``.
        include_general:
            If True, append the ordinary geometry/material ``preflight()``
            issues after the S-parameter routing check.

        Returns
        -------
        PreflightReport
            A ``list`` subclass of :class:`PreflightIssue` (back-compatible with
            the historical ``list[str]``). Carries no ERROR-severity issue when
            the selected calculator is valid for the registered port families —
            read ``report.ok`` / ``report.errors`` rather than emptiness.
            ``calculator="waveguide"`` additionally runs the three setup
            audits (:meth:`_validate_cfg_record_vs_far_boundary`,
            :meth:`_validate_cfg_port_index_mirror_covariance` and
            :meth:`_validate_cfg_layout_from_band_low_edge`), which report
            advisory and informational findings on a perfectly valid routing —
            a healthy two-port guide always carries at least the known
            E-plane-offset note. Those audits are evaluated at
            ``compute_waveguide_s_matrix``'s own default ``num_periods``
            (:data:`WAVEGUIDE_DEFAULT_NUM_PERIODS`); the record-length message
            names the ``num_periods`` that clears the threshold, which does not
            depend on the value it was evaluated at.
        """

        aliases = {
            "run": "run",
            "result": "run",
            "compute_s_params": "run",
            "forward": "forward",
            "forward_s11": "forward",
            "port_s11_freqs": "forward",
            "msl": "msl",
            "compute_msl_s_matrix": "msl",
            "waveguide": "waveguide",
            "compute_waveguide_s_matrix": "waveguide",
            "coaxial": "coaxial",
        }
        if calculator.lower() == "compute_coaxial_s_matrix":
            raise ValueError(
                "calculator='compute_coaxial_s_matrix' names a method that was removed. "
                "Use calculator='coaxial' to check the coaxial methods, "
                "compute_coaxial_line_reflection() (one-port reflection) and "
                "compute_coaxial_two_port() (through line)."
            )
        key = aliases.get(calculator.lower())
        if key is None:
            allowed = ", ".join(sorted(set(aliases.values())))
            raise ValueError(
                f"Unknown S-parameter calculator {calculator!r}. "
                f"Choose one of: {allowed}."
            )

        issues = PreflightReport()

        try:
            if key == "run":
                self._validate_run_sparameter_request(
                    compute_s_params=True,
                    s_param_freqs=None,
                    s_param_n_steps=None,
                    devices=None,
                )
            elif key == "forward":
                self._validate_forward_sparameter_request()
                is_nonuniform = (
                    self._dz_profile is not None
                    or self._dx_profile is not None
                    or self._dy_profile is not None
                )
                if is_nonuniform:
                    raise NotImplementedError(
                        "forward(port_s11_freqs=...) is currently wired only "
                        "on the uniform single-device forward path. Drop "
                        "port_s11_freqs or use a uniform mesh."
                    )
            elif key == "msl":
                self._validate_msl_sparameter_request_for_preflight()
            elif key == "waveguide":
                wg_normalize = False if normalize is None else normalize
                self._validate_waveguide_sparameter_request_for_preflight(
                    normalize=wg_normalize,
                )
            elif key == "coaxial":
                self._validate_coaxial_sparameter_request_for_preflight()
        except (ValueError, NotImplementedError) as exc:
            issues.append(PreflightIssue(
                f"{type(exc).__name__}: {exc}",
                severity="error",
                code=getattr(exc, "code", f"sparam_routing_{key}"),
                source="preflight_sparameters",
                diagnostic=getattr(exc, "diagnostic", None),
            ))

        if key == "msl" and not len(issues):
            if list(self._msl_ports):
                import warnings as _mslmod
                from rfx.preflight.msl import (
                    preflight_msl_probe_clearance,
                )
                with _mslmod.catch_warnings(record=True) as _msl_caught:
                    _mslmod.simplefilter("always")
                    preflight_msl_probe_clearance(
                        self, _mslmod, skip=bool(include_general))
                for _rec in _msl_caught:
                    _inst = _rec.message
                    issues.append(PreflightIssue(
                        _inst,
                        scope=getattr(_inst, "scope", "model"),
                        subjects=getattr(_inst, "_subjects", ()),
                        region=getattr(_inst, "_region", None),
                        severity=getattr(_inst, "severity", "warning"),
                        code=getattr(_inst, "code", "msl_port_geometry"),
                        source=getattr(_inst, "source",
                                       "preflight_sparameters"),
                    ))

        if key == "waveguide" and not len(issues):
            _wg_entries = list(self._waveguide_ports)
            if _wg_entries:
                import warnings as _wmod
                with _wmod.catch_warnings(record=True) as _wg_caught:
                    _wmod.simplefilter("always")
                    self._preflight_waveguide_setup(
                        _wmod,
                        freqs=resolve_waveguide_port_freqs(self, _wg_entries[0]),
                        num_periods=WAVEGUIDE_DEFAULT_NUM_PERIODS,
                    )
                for _rec in _wg_caught:
                    _inst = _rec.message
                    issues.append(PreflightIssue(
                        _inst,
                        scope=getattr(_inst, "scope", "model"),
                        subjects=getattr(_inst, "_subjects", ()),
                        region=getattr(_inst, "_region", None),
                        severity=getattr(_inst, "severity", "warning"),
                        code=getattr(_inst, "code", "uncoded"),
                        loc=getattr(_inst, "loc", None),
                        source=getattr(_inst, "source", None),
                    ))

        if include_general:
            general = self.preflight(strict=False)
            issues.extend(general)
            issues.flux_regions.extend(general.flux_regions)

        from rfx.preflight._subjects import resolve_findings
        resolve_findings(issues)
        _errors = issues.errors
        if strict and _errors:
            _advisory = len(issues) - len(_errors)
            raise issues.exception(
                f"preflight_sparameters (strict) found {len(_errors)} "
                f"error-severity issue(s)"
                + (f" (plus {_advisory} advisory/informational finding(s), "
                   "returned but not escalated)" if _advisory else "")
                + ":\n  - " + "\n  - ".join(_errors)
            )

        if len(issues):
            for issue in issues:
                print(f"  [SPARAM PREFLIGHT] {issue}")
        else:
            print(f"  [SPARAM PREFLIGHT] {key}: all checks passed.")
        return issues

    def _validate_msl_sparameter_request_for_preflight(self) -> None:
        """Mirror ``compute_msl_s_matrix`` family-routing checks."""
        from rfx.preflight.msl_routing import _validate_msl_sparameter_request_for_preflight
        return _validate_msl_sparameter_request_for_preflight(self)

    def _validate_waveguide_sparameter_request_for_preflight(
        self,
        *,
        normalize: bool | str,
    ) -> None:
        """Mirror ``compute_waveguide_s_matrix`` family-routing checks.

        Only the simulation-visible clauses are mirrored (``normalize`` and
        multi-mode on the non-uniform fence); ``compute_waveguide_s_matrix``
        call-time parameters (``subpixel_smoothing``, ``port_reference_sims``,
        ``eps_override`` / ``sigma_override``) are not exposed to preflight and
        stay method-only checks.
        """

        if self._solver != "yee":
            raise NotImplementedError(
                "compute_waveguide_s_matrix() does not support "
                f"solver={self._solver!r}. Use solver='yee'."
            )
        if not self._waveguide_ports:
            raise ValueError(
                "No waveguide ports registered. Call add_waveguide_port() first."
            )
        if self._ports or self._tfsf:
            raise ValueError(
                "compute_waveguide_s_matrix() is not supported together with "
                "a lumped or wire port, or a soft source (add_source) or TFSF"
            )
        if self._periodic_axes:
            raise ValueError(
                "compute_waveguide_s_matrix() is not supported with manual "
                "periodic-axis overrides"
            )
        if len(self._waveguide_ports) < 2:
            raise ValueError(
                "compute_waveguide_s_matrix() requires at least two "
                "waveguide ports"
            )

        entries = list(self._waveguide_ports)
        if any(entry.probe_plane is not None for entry in entries):
            raise ValueError(
                "compute_waveguide_s_matrix() does not use per-port "
                "probe_plane; use reference_plane only or leave probe_plane unset"
            )
        if any(entry.calibration_preset not in (None, "measured") for entry in entries):
            raise ValueError(
                "compute_waveguide_s_matrix() currently supports only "
                "measured/default reference planes or explicit reference_plane "
                "overrides"
            )
        if (
            self._dz_profile is not None
            or self._dx_profile is not None
            or self._dy_profile is not None
        ):
            unsupported = []
            if normalize is not True and normalize != "flux":
                unsupported.append("normalize=True or normalize='flux' is required")
            if any(entry.n_modes > 1 for entry in entries):
                unsupported.append("multi-mode ports (n_modes>1) are not supported")
            if unsupported:
                raise NotImplementedError(
                    "compute_waveguide_s_matrix() on a non-uniform mesh "
                    "(dx_profile / dy_profile / dz_profile) supports "
                    "normalize=True or "
                    "normalize='flux' and single-mode ports. "
                    + "; ".join(unsupported)
                    + ". Drop the dx/dy/dz profile to use the uniform lane."
                )

    def _validate_coaxial_sparameter_request_for_preflight(self) -> None:
        """The registration checks both coaxial lanes make before any solve.

        ``compute_coaxial_line_reflection`` and ``compute_coaxial_two_port``
        refuse the same registrations, in the same words, before they build
        a grid (``rfx/sparams/coax.py``). Each condition below is one of
        those refusals, read off the lanes' own guard blocks; every failing
        one is listed, so a setup is fixed in one pass. What the lanes check
        on their METHOD arguments (``termination``, ``probe_count``, whether
        the probe ladder fits the domain) cannot be seen here.
        ``tests/unit/preflight/test_removed_coaxial_s_matrix_lane.py`` runs
        this check and both lanes on the same setups and requires the three
        to agree.
        """

        if not self._coaxial_ports:
            raise ValueError(
                "No coaxial ports registered. Call add_coaxial_port() first."
            )

        failed: list[str] = []
        n_coax = len(self._coaxial_ports)
        if n_coax != 1:
            failed.append(
                f"exactly one add_coaxial_port() ({n_coax} registered)")
        others = [
            name for name, entries in (
                ("add_port", self._ports),
                ("add_waveguide_port", self._waveguide_ports),
                ("add_floquet_port", self._floquet_ports),
                ("add_msl_port", self._msl_ports),
            ) if entries
        ]
        if others:
            failed.append(
                "no port family besides add_coaxial_port() (registered: "
                + ", ".join(others) + ")")
        faces = sorted({str(port.face) for port in getattr(self, "_coaxial_ports", ())})
        if faces != ["top"]:
            failed.append(f"face='top' (registered: {', '.join(faces)})")
        if self._boundary != "cpml" or self._cpml_layers <= 0:
            failed.append(
                "boundary='cpml' with cpml_layers > 0 (got "
                f"boundary={self._boundary!r}, cpml_layers={self._cpml_layers})")
        else:
            from rfx.boundaries.depths import has_positive_cpml_faces
            if not has_positive_cpml_faces(self._boundary_spec, self._cpml_layers):
                failed.append("positive CPML thickness on all six faces")
            if any(token != "cpml"
                   for _, _, token in self._boundary_spec.faces()):
                failed.append("CPML on all six boundary faces")
        if self._periodic_axes:
            failed.append("no periodic boundary axes")
        if self._mode != "3d":
            failed.append(f"mode='3d' (got {self._mode!r})")
        if self._solver != "yee":
            failed.append(f"solver='yee' (got {self._solver!r})")
        if self._precision != "float32":
            failed.append(f"precision='float32' (got {self._precision!r})")
        if self._stencil_order != 2:
            failed.append(f"stencil_order=2 (got {self._stencil_order})")
        if self._tfsf is not None:
            failed.append("no TFSF source (each method builds its own)")
        declared_mesh = self._declared_mesh
        if any(declared_mesh[name] is not None
               for name in ("_dx_profile", "_dy_profile", "_dz_profile")):
            failed.append("a uniform grid (no dx/dy/dz profile)")
        if self._refinement is not None:
            failed.append("no SBP-SAT refinement")
        if self._geometry or self._thin_conductors:
            failed.append(
                "no registered geometry or thin conductors (each method "
                "builds the complete line)")
        if self._lumped_rlc:
            failed.append("no lumped RLC elements")
        if self._probes or self._dft_planes or self._flux_monitors or self._ntff:
            failed.append(
                "no registered probes, DFT planes, flux monitors or NTFF boxes")
        if failed:
            raise ValueError(
                "compute_coaxial_line_reflection() and compute_coaxial_two_port() "
                "both refuse this setup. They require: " + "; ".join(failed)
                + ".")

    from rfx.preflight.ntff import (
        _validate_ntff_inverse_design,
        _validate_ntff_small_ground_plane,
    )

    from rfx.preflight.f0_sheet_backed_gap import _validate_cfg_f0_sheet_backed_gap

    def _validate_simulation_config(self) -> None:
        """Comprehensive pre-simulation configuration validation.

        Checks for common setup mistakes that produce silent wrong results:
        probe/source in CPML, boundary type mismatch, feature compatibility,
        NTFF precision, normalize defaults.

        Called from run() after _validate_mesh_quality().

        Stage 1b refactor: the original ~592-line body was
        decomposed into per-check ``_validate_cfg_*`` helpers. This method
        keeps its signature and remains the public entry point; its body
        computes the shared local state (``dx``, CPML thicknesses,
        ``absorber_label``) and then calls each helper IN THE SAME ORDER
        as the original checks. No logic, ordering, or warning text
        changed — pure readability decomposition.

        The helper calls are registered in ``rfx/preflight/_registry.py`` as
        :data:`~rfx.preflight._registry.CORE_CONFIG_CHECKS`, one entry per
        call, in the same order and with the same arguments — so a new check
        is registered in its family module instead of edited into this body,
        and the suite can be listed without reading it. What stays here is
        exactly the context-building step: this method still computes the
        shared state once and hands it over as a
        :class:`~rfx.preflight._registry.ConfigCheckContext`.
        """
        from rfx.preflight._registry import ConfigCheckContext, run_config_checks

        import warnings as _w

        dx = self._dx or C0 / self._freq_max / 20.0
        cpml_thickness = self._cpml_layers * dx if self._boundary in ("cpml", "upml") else 0

        cpml_thick_lo, cpml_thick_hi, _pmc_faces_set = (
            self._validate_cfg_compute_cpml_thickness(cpml_thickness)
        )
        absorber_label = "UPML" if self._boundary == "upml" else "CPML"

        run_config_checks(self, ConfigCheckContext(
            warn=_w,
            dx=dx,
            cpml_thickness=cpml_thickness,
            cpml_thick_lo=cpml_thick_lo,
            cpml_thick_hi=cpml_thick_hi,
            pmc_faces=_pmc_faces_set,
            absorber_label=absorber_label,
        ))

    def _validate_cfg_precision_x64(self, _w) -> None:
        """Report missing x64 as an error; retain the nonuniform advisory."""
        from rfx._internal._precision import FLOAT64_MODE_ERROR

        if self._precision == "float64" and not jax.config.x64_enabled:
            _w.warn(PreflightWarning(
                FLOAT64_MODE_ERROR,
                severity="error",
                code="precision_float64_without_x64",
                source="_validate_cfg_precision_x64",
            ))

        is_nonuniform = (
            self._dz_profile is not None
            or self._dx_profile is not None
            or self._dy_profile is not None
        )
        if self._precision == "mixed" and is_nonuniform:
            _w.warn(PreflightWarning(
                f"precision={self._precision!r} was requested on a "
                "non-uniform mesh (dx/dy/dz profile set), but the "
                "non-uniform runner does not thread field_dtype -- fields "
                "would silently run float32 regardless of this setting"
                ". run()/forward() will raise NotImplementedError "
                "at dispatch rather than proceed silently; this warning "
                "exists to explain that error before you hit it. Use "
                "precision='float32' (the default) with a non-uniform mesh, "
                "or drop the mesh profile to reach the uniform lane where "
                "this precision knob is currently supported.",
                code="precision_nonuniform_lane_unsupported",
                source="_validate_cfg_precision_x64",
            ))


    from rfx.preflight.mesh import _validate_cfg_graded_box_rasterization

    from rfx.preflight.ports import _validate_cfg_refplane_placement







    from rfx.preflight.absorber import (
        _validate_cfg_lossless_resonator_in_absorber,
        _validate_cfg_dispersive_pole_at_absorber_face,
        _validate_cfg_compute_cpml_thickness,
        _validate_cfg_absorber_budget_vs_grid,
        _preflight_face_layers,
        _validate_cfg_pec_faces_with_finite_pec,
        _validate_cfg_upml_refinement,
        _validate_cfg_upml_nonuniform_lane,
        _validate_cfg_dielectric_at_absorber_seam,
        _validate_cfg_port_conductor_continues,
        _validate_cfg_conductor_in_thin_absorber,
        _validate_cfg_thin_absorber,
    )

    from rfx.preflight.mesh import _validate_cfg_floquet_nonuniform

    from rfx.preflight.absorber import _validate_cfg_absorber_placement

    from rfx.preflight.sources import _validate_cfg_source_on_reflector_plane

    from rfx.preflight.ntff import _validate_cfg_ntff_absorber_overlap

    def _validate_cfg_settling_witness_present(self, _w) -> None:
        """Warn when the declared inputs cannot produce a ring-down witness.

        Input-side fact, not a result prediction: ``run()`` and ``forward()`` score their
        energy ring-down settling witness from the probe time series,
        so a simulation that registers NTFF or a field-DFT plane and NO
        point probe will come back with ``settling_db=None`` -- the
        claims-bearing open-domain DFT numbers this project's -40 dB
        settling rule governs, with the truncation guard absent. Cheaper to
        say here than after the run.

        Silent when a probe exists, and silent when the run asks for neither
        NTFF nor a field DFT (the rule scopes to those). Whether a
        registered probe is INDEPENDENT of the drive is decided on the
        finished record, not here: the run/forward witness flags a probe
        sharing a Yee cell with a registered drive as source-dominated and
        says so in ``settling_witness["qualifier"]`` plus a runtime warning. This advisory only states the advice,
        which is why its
        text names the drive cell.

        Both entry points use the same recorded-probe arithmetic. Forward's
        host diagnostic is available on concrete results, including after
        an outer JIT returns; while records are traced it is unavailable.
        A registered probe does not imply coverage if its recording is
        disabled, too short, or below the storage-format underflow floor.
        """
        if self._ntff is None and not self._dft_planes:
            return
        if self._probes:
            return
        wants = []
        if self._ntff is not None:
            wants.append("NTFF far-field output")
        if self._dft_planes:
            wants.append("field-DFT plane probe(s)")
        _w.warn(PreflightWarning(
            f"this simulation requests {' and '.join(wants)} but registers no "
            "point probe. run() and forward() score their ring-down settling witness from "
            "the recorded probe time series, so the result will carry settling_db=None "
            "and those DFT numbers will have no truncation guard; add "
            "sim.add_probe(position, component) somewhere the field is live "
            "and NOT on a source/port drive cell — a record on the drive "
            "cell peaks on the drive pulse, so its end/peak ratio measures "
            "source turn-off rather than the structure's ring-down and does "
            "not count as an independent witness — and retain its "
            "time series. Inspect forward's diagnostic on "
            "the concrete result after JIT/AD evaluation.",
            code="settling_witness_will_be_absent",
            source="_validate_cfg_settling_witness_present",
        ))

    from rfx.preflight.absorber import _validate_cfg_geometry_in_cpml, _validate_cfg_body_flush_on_absorber

    from rfx.preflight.ports import (
        _validate_cfg_port_inside_pec,
        _wire_port_cell_centers,
        _validate_cfg_floating_single_cell_port,
        _validate_cfg_half_node_split,
    )

    from rfx.preflight.absorber import _validate_cfg_pec_boundary_open_structure

    from rfx.preflight.sources import _validate_cfg_no_sources

    _MULTIBAND_RATIO_CAP = 1.4
    _INPLANE_RATIO_CAP = 1.4

    from rfx.preflight.mesh import (
        _validate_cfg_multiband_grading,
        _validate_cfg_thin_conductor_graded_node,
    )


    _AXIS_OF_COMPONENT = {"ex": 0, "ey": 1, "ez": 2}

    from rfx.preflight.mesh import (
        _graded_node_report,
        _validate_cfg_source_on_graded_node,
        _validate_cfg_wire_port_on_graded_node,
    )

    from rfx.preflight.pec_geometry import (
        _validate_cfg_thin_conductor_surface_impedance,
        _validate_cfg_thin_conductor_thickness,
    )


    from rfx.preflight.sources import _validate_cfg_unresolved_pulse

    from rfx.preflight.mesh import (
        _validate_cfg_nonuniform_limitations,
        _validate_cfg_subgrid_limitations,
    )

    from rfx.preflight.waveguide import _validate_cfg_waveguide_reference_plane
    from rfx.preflight.waveguide_port_span import _validate_cfg_waveguide_port_span

    from rfx.preflight.waveguide import (
        _waveguide_setup_planes,
        _waveguide_far_geometry,
        _validate_cfg_layout_from_band_low_edge,
        _validate_cfg_record_vs_far_boundary,
        _validate_cfg_port_index_mirror_covariance,
        _preflight_waveguide_setup,
    )

    from rfx.preflight.msl import (
        _msl_assemble_once,
        _msl_declared_face_geometry,
        _msl_realized_substrate,
        _msl_conductor_gap,
        _check_msl_port_geometry,
    )


    from rfx.preflight.realization import _campaign_ctx, _validate_cfg_assembly, clear_inspection_cache
    from rfx.preflight.pec_geometry import (
        _validate_cfg_campaign_statics,
        _validate_cfg_pec_realization,
        _validate_cfg_sheet_slot_vacuum,
        _validate_cfg_pec_face_short_of_domain_wall,
        _congruence_origin_shift,
        _validate_cfg_congruent_rasterization_parity,
        _validate_cfg_sheet_cavity_thickness,
        _validate_cfg_off_lattice_design_edges,
    )

    _congruence_origin_shift = staticmethod(_congruence_origin_shift)


