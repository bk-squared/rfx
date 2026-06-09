"""Preflight and configuration-validation methods for :class:`Simulation`.

Import contract (Part B Stage 1a refactor):
  This module is a transitional mixin. It must import ONLY from
  ``rfx.api._spec`` plus external ``rfx.*`` / stdlib / jax / numpy.
  It must NEVER do ``from rfx.api import ...`` or ``from . import ...``
  the package, to keep ``rfx/api/__init__.py`` the sole composition point.

The methods here were moved verbatim out of ``rfx/api/__init__.py``'s
``class Simulation`` body. They are pure structural relocations — same
indentation, decorators, signatures, and logic. ``Simulation`` inherits
``_PreflightMixin`` so every method below remains a bound method on
``Simulation`` instances; ~79 test call-sites are unaffected.
"""

from __future__ import annotations

import json
import math

import jax.numpy as jnp
import numpy as np

from rfx.grid import C0
from rfx.core.yee import MaterialArrays
from rfx.core.jax_utils import is_tracer


class PreflightWarning(UserWarning):
    """Base for structured preflight findings carried on the warning instance.

    Mirrors the in-repo report idioms (:class:`SubgridValidationIssue`,
    :class:`PortValidationIssue`): the check site sets a stable lowercase-slug
    ``code`` and a ``severity`` on the warning instance, plus optional ``loc``
    (where in the setup the finding applies) and ``source`` (the check method
    name). ``preflight()`` reads these fields off ``w.message`` so the issue
    record is coded at the check site rather than inferred from text.

    Emit with ``warnings.warn(PreflightWarning(msg, code="...", source="..."))``.
    """

    def __init__(
        self,
        message,
        *,
        code: str = "uncoded",
        severity: str = "warning",
        loc: str | None = None,
        source: str | None = None,
    ) -> None:
        super().__init__(message)
        self.message = str(message)
        self.code = code
        self.severity = severity
        self.loc = loc
        self.source = source

    def __str__(self) -> str:  # back-compat: warning prints as its message
        return self.message


class PreflightErrorWarning(PreflightWarning):
    """An error-severity preflight finding emitted as a warning.

    Re-parented under :class:`PreflightWarning` (Phase A). Emitting (rather than
    raising) keeps the rest of the preflight suite running so the user sees ALL
    issues at once, while ``preflight()`` still tags the resulting
    :class:`PreflightIssue` with ``severity="error"`` so an automation agent can
    gate on it. Use for known-bad configurations that should stop a run.

    ``severity`` defaults to ``"error"``; the legacy
    ``warnings.warn("msg", PreflightErrorWarning)`` form (category, no instance
    attrs) still surfaces as error-severity via ``preflight()``'s
    ``issubclass(w.category, PreflightErrorWarning)`` derivation.
    """

    def __init__(
        self,
        message,
        *,
        code: str = "uncoded",
        severity: str = "error",
        loc: str | None = None,
        source: str | None = None,
    ) -> None:
        super().__init__(
            message, code=code, severity=severity, loc=loc, source=source
        )


class PreflightConfigError(ValueError):
    """A structurally-impossible-config raise carrying a check-site ``code``.

    The structurally-impossible config validators (``upml``+refinement,
    Floquet+non-uniform-z, ...) raise this so ``preflight()`` can record the
    error-severity :class:`PreflightIssue` with the slug set at the check site
    instead of inferring it from the message. It subclasses ``ValueError`` so
    every existing ``except ValueError`` / ``pytest.raises(ValueError)`` site
    (including the run() regression locks) is unaffected.
    """

    def __init__(
        self,
        message,
        *,
        code: str = "uncoded",
        loc: str | None = None,
        source: str | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.loc = loc
        self.source = source


class PreflightIssue(str):
    """One preflight finding.

    Subclasses ``str`` so it is 100% back-compatible with the plain
    ``list[str]`` that ``preflight()`` has always returned — it prints, joins,
    compares, and regex-matches exactly like its message — while also carrying
    machine-readable fields so an automation agent can gate deterministically::

        report = sim.preflight()
        errors = [i for i in report if i.severity == "error"]
        if errors:
            ...  # stop before spending GPU minutes on a doomed run

    ``severity`` is ``"error"`` for hard contradictions / known-bad configs and
    ``"warning"`` for advisories. ``code`` is the lowercase-slug category set at
    the check site (e.g. ``"conformal_nan"``, ``"mesh_resolution"``,
    ``"absorber_overlap"``). ``loc`` and ``source`` are optional provenance.

    The str subclass silently drops these attrs under ``json.dumps`` — never
    serialize a :class:`PreflightIssue` directly; use :meth:`to_dict` or the
    owning :class:`PreflightReport`'s :meth:`PreflightReport.to_dict` /
    :meth:`PreflightReport.to_json`.
    """

    severity: str
    code: str
    loc: str | None
    source: str | None

    def __new__(
        cls,
        message,
        *,
        severity: str = "warning",
        code: str = "uncoded",
        loc: str | None = None,
        source: str | None = None,
    ):
        obj = super().__new__(cls, str(message))
        obj.severity = severity
        obj.code = code
        obj.loc = loc
        obj.source = source
        return obj

    def to_dict(self) -> dict[str, object]:
        """Return a stable, JSON-serializable record of this finding."""
        return {
            "message": str(self),
            "code": self.code,
            "severity": self.severity,
            "loc": self.loc,
            "source": self.source,
        }


class PreflightReport(list):
    """Structured result of :meth:`Simulation.preflight`.

    A ``list`` subclass holding :class:`PreflightIssue` items, so it IS a list
    and every legacy ``list[str]`` call site (iterate / ``"\\n".join`` / ``len``
    / truthiness) keeps working unchanged. It also exposes the canonical report
    API shared with :class:`rfx.validation.PortValidationReport` and
    :class:`rfx.subgridding.validation.SubgridValidationReport`.
    """

    @property
    def issues(self) -> list:
        """All findings as a plain list (mirrors the other report classes)."""
        return list(self)

    @property
    def errors(self) -> list:
        """Error-severity findings only."""
        return [i for i in self if getattr(i, "severity", "warning") == "error"]

    @property
    def warnings(self) -> list:
        """Non-error (advisory) findings only."""
        return [i for i in self if getattr(i, "severity", "warning") != "error"]

    @property
    def ok(self) -> bool:
        """Whether the report contains no error-severity finding."""
        return not self.errors

    def by_code(self, code: str) -> list:
        """Return all findings with diagnostic ``code``."""
        return [i for i in self if getattr(i, "code", None) == code]

    def format(self) -> str:
        """Return a compact human-readable multiline summary."""
        status = "PASS" if self.ok else "FAIL"
        if not self:
            return f"preflight: {status} (no issues)"
        lines = [f"preflight: {status} ({len(self)} issue(s))"]
        for issue in self:
            sev = getattr(issue, "severity", "warning")
            code = getattr(issue, "code", "uncoded")
            lines.append(f"- {sev.upper()} [{code}] {issue}")
        return "\n".join(lines)

    def raise_for_failure(self) -> "PreflightReport":
        """Raise ``ValueError`` listing every error-severity finding.

        Returns ``self`` on success so callers can use it as both a fail-fast
        gate and an artifact (the R3 pre-VESSL gate). No-op when :attr:`ok`.
        """
        errors = self.errors
        if errors:
            detail = "\n  - ".join(str(e) for e in errors)
            raise ValueError(
                f"preflight found {len(errors)} blocking error(s):\n  - {detail}"
            )
        return self

    def to_dict(self) -> dict[str, object]:
        """Return a stable, JSON-serializable validation artifact.

        Real serialization (unlike ``json.dumps`` of a bare
        :class:`PreflightIssue`, which drops the code/severity attrs).
        """
        return {
            "ok": self.ok,
            "n_issues": len(self),
            "n_errors": len(self.errors),
            "issues": [
                i.to_dict() if isinstance(i, PreflightIssue)
                else {
                    "message": str(i),
                    "code": getattr(i, "code", "uncoded"),
                    "severity": getattr(i, "severity", "warning"),
                    "loc": getattr(i, "loc", None),
                    "source": getattr(i, "source", None),
                }
                for i in self
            ],
        }

    def to_json(self, **kwargs: object) -> str:
        """Serialize the report for research-note artifacts."""
        options = {"indent": 2, "sort_keys": True}
        options.update(kwargs)
        return json.dumps(self.to_dict(), **options)


class _PreflightMixin:
    """Preflight / validation methods mixed into :class:`Simulation`."""

    @staticmethod
    def _validate_tfsf_vacuum_boundary(materials: MaterialArrays, tfsf_cfg) -> None:
        """Ensure the TFSF x-boundary planes remain vacuum.

        The current TFSF correction assumes vacuum on and immediately
        adjacent to the TFSF x boundaries. Fail loudly instead of
        allowing silently wrong scattered fields.
        """
        boundary_slices = (
            ("x_lo-1", slice(tfsf_cfg.x_lo - 1, tfsf_cfg.x_lo)),
            ("x_lo", slice(tfsf_cfg.x_lo, tfsf_cfg.x_lo + 1)),
            ("x_hi", slice(tfsf_cfg.x_hi, tfsf_cfg.x_hi + 1)),
            ("x_hi+1", slice(tfsf_cfg.x_hi + 1, tfsf_cfg.x_hi + 2)),
        )

        for plane_name, xs in boundary_slices:
            eps = np.asarray(materials.eps_r[xs, :, :])
            sigma = np.asarray(materials.sigma[xs, :, :])
            mu = np.asarray(materials.mu_r[xs, :, :])
            if not (
                np.allclose(eps, 1.0)
                and np.allclose(sigma, 0.0)
                and np.allclose(mu, 1.0)
            ):
                raise ValueError(
                    "TFSF plane-wave source requires vacuum on and adjacent to "
                    f"the TFSF x boundaries; non-vacuum material found at {plane_name}"
                )

    def _validate_run_sparameter_request(
        self,
        *,
        compute_s_params: bool | None,
        s_param_freqs,
        s_param_n_steps: int | None,
        devices: list | None = None,
    ) -> None:
        """Reject explicit ``run`` S-parameter requests outside its contract."""

        requested = (
            compute_s_params is True
            or s_param_freqs is not None
            or s_param_n_steps is not None
        )
        if not requested:
            return

        port_entries = self._port_sparameter_entries()
        source_only_entries = [pe for pe in self._ports if pe.impedance == 0.0]
        messages: list[str] = []

        if self._msl_ports:
            messages.append(
                "add_msl_port(...) uses compute_msl_s_matrix(); "
                "run(compute_s_params=True) does not include MSL ports in "
                "Result.s_params"
            )
        if self._waveguide_ports:
            messages.append(
                "add_waveguide_port(...) uses compute_waveguide_s_matrix() "
                "for the full S-matrix; run() may return per-port "
                "result.waveguide_sparams but not Result.s_params"
            )
        if self._floquet_ports:
            messages.append(
                "add_floquet_port(...) is experimental and has no "
                "claims-bearing run(compute_s_params=True) S-matrix path"
            )
        if self._tfsf is not None:
            messages.append(
                "add_tfsf_source(...) is a plane-wave source, not a port"
            )
        if self._coaxial_ports:
            messages.append(
                "add_coaxial_port(...) is not wired into run(compute_s_params=True); "
                "use Simulation.compute_coaxial_s_matrix(...) (experimental TEM "
                "plane-source API) or add_port(extent=...) for the current "
                "probe-feed S-parameter path"
            )

        if not port_entries:
            if source_only_entries:
                messages.append(
                    "add_source(...) / add_polarized_source(...) are "
                    "source-only observables and cannot populate "
                    "Result.s_params"
                )
            detail = "; ".join(messages) if messages else (
                "register at least one add_port(...) impedance port"
            )
            raise ValueError(
                "run(compute_s_params=True) computes Result.s_params only "
                f"for add_port(...) lumped or wire ports; {detail}."
            )

        if messages:
            raise NotImplementedError(
                "run(compute_s_params=True) has a single result schema for "
                "add_port(...) lumped/wire ports. Mixed or specialized port "
                "families must use their documented calculators: "
                + "; ".join(messages)
                + "."
            )

        if self._solver == "adi":
            raise NotImplementedError(
                "run(compute_s_params=True) is not supported with "
                "solver='adi'; use the uniform Yee solver."
            )
        if devices is not None and len(devices) > 1:
            raise NotImplementedError(
                "run(compute_s_params=True) is not supported on the "
                "distributed multi-device path; run a single-device "
                "uniform S-parameter calculation."
            )
        if self._refinement is not None:
            if source_only_entries:
                raise NotImplementedError(
                    "subgrid compute_s_params ignores ordinary "
                    "add_source(...) entries like the uniform S-matrix "
                    "extractor; remove source-only entries and drive through "
                    "add_port(...) waveforms."
                )
            if any(pe.waveform is None for pe in port_entries):
                raise ValueError(
                    "subgrid compute_s_params needs a waveform "
                    "on every impedance port so each port can be driven in "
                    "turn. Pass waveform=... even for ports whose main-run "
                    "excite flag is False."
                )

        is_nonuniform = (
            self._dz_profile is not None
            or self._dx_profile is not None
            or self._dy_profile is not None
        )
        if is_nonuniform and any(pe.extent is None for pe in port_entries):
            raise NotImplementedError(
                "run(compute_s_params=True) on a non-uniform mesh is wired "
                "only for add_port(..., extent=...) WirePort extraction. "
                "Single-cell lumped-port S-parameters require the uniform "
                "reference lane."
            )

    def _validate_forward_sparameter_request(self) -> None:
        """Reject ``forward(port_s11_freqs=...)`` outside its narrow path."""

        port_entries = self._port_sparameter_entries()
        messages: list[str] = []
        if self._msl_ports:
            messages.append("MSL ports use compute_msl_s_matrix()")
            # Near-field guard (issue #80): probe 0 must clear the source
            # FRINGING transient (~5·h_sub), which decays over a few substrate
            # thicknesses, not over λ. Inside it the V·I-split S11 of a high-Q
            # resonant load is corrupted (the issue-#80 edge-fed patch read
            # |S11|=8.94/1.11 at offset~5; passive ~0.99 once cleared). The
            # default n_probe_offset already floors to max(λ-clearance,
            # 5·h_sub/dx); this warns when an EXPLICIT value under-provisions it.
            for pe in self._msl_ports:
                if self._dx and pe.n_probe_offset < 5.0 * pe.height / self._dx:
                    messages.append(
                        f"MSL port {pe.name!r}: n_probe_offset="
                        f"{pe.n_probe_offset} sits within the source fringing "
                        f"transient (~{5.0 * pe.height / self._dx:.0f} cells = "
                        f"5·h_sub/dx); probe 0 may corrupt the V·I-split S11 of "
                        f"a high-Q resonant load (issue #80) — increase "
                        f"n_probe_offset or leave it None for the safe default."
                    )
        if self._waveguide_ports:
            messages.append("waveguide ports use compute_waveguide_s_matrix()")
        if self._floquet_ports:
            messages.append(
                "Floquet ports are experimental and have no forward S11 path"
            )
        if self._tfsf is not None:
            messages.append("TFSF is a plane-wave source, not a port")
        if self._coaxial_ports:
            messages.append(
                "coaxial ports are not wired into forward(port_s11_freqs=...); "
                "use Simulation.compute_coaxial_s_matrix(...) for the "
                "experimental coaxial S-matrix path"
            )
        if not port_entries:
            source_only = any(pe.impedance == 0.0 for pe in self._ports)
            if source_only:
                messages.append("add_source(...) is not an impedance port")
            detail = "; ".join(messages) if messages else (
                "register add_port(...) first"
            )
            raise ValueError(
                "forward(port_s11_freqs=...) computes S11 only for "
                f"add_port(...) lumped or wire ports on the uniform "
                f"single-device path; {detail}."
            )
        if messages:
            raise NotImplementedError(
                "forward(port_s11_freqs=...) cannot be combined with "
                "specialized or non-port excitation families: "
                + "; ".join(messages)
                + "."
            )

    def _validate_mesh_quality(self) -> None:
        """Pre-simulation mesh quality check (P0).

        Scans all geometry elements against the grid cell size and warns
        about under-resolved features. Prevents silent garbage results
        from mesh-related setup errors.
        """
        import warnings as _w

        # Tracer-valued profiles (mesh-as-design-variable gradient) cannot
        # participate in host-side min/len/indexing. Advisory warnings
        # are skipped in that case — correctness is preserved downstream.
        if any(
            p is not None and is_tracer(p)
            for p in (self._dx_profile, self._dy_profile, self._dz_profile)
        ):
            return

        dx = self._dx
        if dx is None:
            dx = C0 / self._freq_max / 20.0

        # Determine minimum cell size per axis — use profile min when
        # non-uniform xy is active, so we don't flag features that are
        # actually well-resolved in their local fine-mesh region.
        min_dx = float(min(self._dx_profile)) if self._dx_profile is not None else dx
        min_dy = float(min(self._dy_profile)) if self._dy_profile is not None else dx
        if self._dz_profile is not None:
            min_dz = min(self._dz_profile)
        else:
            min_dz = dx

        for entry in self._geometry:
            shape = entry.shape
            mat_name = entry.material_name

            # Get bounding box dimensions
            if hasattr(shape, "bounding_box"):
                try:
                    c1, c2 = shape.bounding_box()
                    dims = [abs(c2[i] - c1[i]) for i in range(3)]
                except (NotImplementedError, TypeError):
                    continue
            else:
                continue

            cell_sizes = [min_dx, min_dy, min_dz]

            # FP1 refinement (2026-05-06): the partial-volume warning
            # at 3-5 cells along one axis is meaningful only for actual
            # *volumes* (≥3 cells in every axis).  A thin strip
            # (e.g. an MSL trace at LX × W_trace × dx → many × 4.7 × 1
            # cells) is a sheet, not a volume, and the per-axis 4.7
            # signal must not fire.  Compute cells on every axis up
            # front and gate the volume branch on the minimum.
            cells_per_axis = [
                (dim / cell) if cell > 0 else float("inf")
                for dim, cell in zip(dims, cell_sizes)
            ]
            is_thin_along_some_axis = min(cells_per_axis) < 3.0

            for axis, (dim, cell) in enumerate(zip(dims, cell_sizes)):
                if dim <= 0:
                    # Zero-thickness geometry
                    axis_name = "xyz"[axis]
                    _w.warn(
                        PreflightWarning(
                            f"Zero-thickness geometry '{mat_name}' along "
                            f"{axis_name}-axis. On non-uniform mesh this may "
                            f"produce empty rasterization. Consider giving it "
                            f"at least one cell of thickness ({cell*1e3:.2f}mm).",
                            code="mesh_resolution",
                            source="_validate_mesh_quality",
                        ),
                        stacklevel=3,
                    )
                elif dim < cell:
                    axis_name = "xyz"[axis]
                    cells_count = dim / cell
                    # Check if this is a PEC material that could use thin sheet
                    mat = self._resolve_material(mat_name)
                    is_pec = mat.sigma >= self._PEC_SIGMA_THRESHOLD
                    hint = (
                        " Use add_thin_conductor() for sub-cell PEC sheet."
                        if is_pec else
                        " Use non-uniform mesh or reduce dx."
                    )
                    _w.warn(
                        PreflightWarning(
                            f"'{mat_name}' {axis_name}-extent {dim*1e3:.2f}mm = "
                            f"{cells_count:.1f} cells — below 1 cell resolution."
                            + hint,
                            code="mesh_resolution",
                            source="_validate_mesh_quality",
                        ),
                        stacklevel=3,
                    )
                else:
                    # Physics-based resolution thresholds (issue #37).
                    # PEC with extent <3 cells is a thin sheet — 1-cell
                    # rasterization is canonical. Only warn on partial
                    # volume: 3-5 cells thick PEC slabs.
                    # Dielectric: cells per local λ_eff, not cells per
                    # geometry extent.
                    mat = self._resolve_material(mat_name)
                    is_pec = mat.sigma >= self._PEC_SIGMA_THRESHOLD
                    axis_name = "xyz"[axis]
                    cells = dim / cell
                    if is_pec:
                        if (3.0 <= cells < 5.0
                                and not is_thin_along_some_axis):
                            _w.warn(
                                PreflightWarning(
                                    f"PEC '{mat_name}' {axis_name}-extent "
                                    f"{dim*1e3:.2f}mm = {cells:.1f} cells — "
                                    "volume under-resolved (PEC volume needs "
                                    "≥5 cells; thin sheets <3 cells are fine).",
                                    code="mesh_resolution",
                                    source="_validate_mesh_quality",
                                ),
                                stacklevel=3,
                            )
                    else:
                        eps_r = float(mat.eps_r) if mat.eps_r else 1.0
                        lam_eff = (
                            C0 / self._freq_max / math.sqrt(max(eps_r, 1.0))
                        )
                        cells_per_lam = lam_eff / cell
                        # rfx's Yee update is 2nd-order in bulk but
                        # degrades to 1st-order at ε-discontinuities
                        # because subpixel smoothing is default OFF
                        # (Meep ships it ON and stays 2nd-order). For
                        # phase-accurate propagation we need ≥15 cells
                        # per λ_eff — the traditional λ/10 rule applies
                        # to subpixel-smoothed codes. S-parameter
                        # extraction with a port or flux monitor
                        # amplifies dielectric-interface phase error
                        # into |S| magnitude error (see
                        # examples/crossval/11 rfx-vs-analytic audit,
                        # 2026-04-24): at 17.7 cells/λ_eff we measure
                        # ~5% |S21| deficit at Fabry-Perot peaks; at
                        # 35 cells/λ_eff (dx halved) it halves to ~2%.
                        # Require 20 cells/λ_eff when S-param
                        # extraction is active.
                        sparam_active = bool(
                            self._waveguide_ports
                            or self._flux_monitors
                        )
                        threshold = 20.0 if sparam_active else 15.0
                        if cells_per_lam < threshold:
                            suffix = (
                                " S-parameter extraction amplifies "
                                "ε-interface phase error into |S| "
                                "magnitude error; ~5% |S21| deficit "
                                "expected at 17 cells/λ_eff."
                                if sparam_active else
                                " Yee without subpixel smoothing has "
                                "1st-order convergence at ε interfaces."
                            )
                            _w.warn(
                                PreflightWarning(
                                    f"dielectric '{mat_name}' on {axis_name}: "
                                    f"{cells_per_lam:.1f} cells per λ_eff "
                                    f"(eps_r={eps_r:.2f}, freq_max="
                                    f"{self._freq_max/1e9:.2f}GHz, "
                                    f"dx={cell*1e3:.3f}mm). Need ≥"
                                    f"{threshold:.0f} cells/λ_eff for "
                                    f"phase-accurate propagation."
                                    f"{suffix}",
                                    code="mesh_resolution",
                                    source="_validate_mesh_quality",
                                ),
                                stacklevel=3,
                            )

        # Check gaps between PEC structures
        pec_entries = [e for e in self._geometry if e.material_name == "pec"]
        if len(pec_entries) >= 2:
            for i in range(len(pec_entries)):
                for j in range(i + 1, min(i + 5, len(pec_entries))):
                    try:
                        c1a, c2a = pec_entries[i].shape.bounding_box()
                        c1b, c2b = pec_entries[j].shape.bounding_box()
                        # Min gap along each axis
                        for ax in range(3):
                            gap = max(0, max(c1b[ax] - c2a[ax], c1a[ax] - c2b[ax]))
                            cell = [dx, dx, min_dz][ax]
                            if 0 < gap < 3 * cell:
                                _w.warn(
                                    PreflightWarning(
                                        f"Gap between PEC structures: "
                                        f"{gap*1e3:.2f}mm = {gap/cell:.1f} cells "
                                        f"along {'xyz'[ax]} — coupling may be "
                                        f"under-resolved.",
                                        code="mesh_resolution",
                                        source="_validate_mesh_quality",
                                    ),
                                    stacklevel=3,
                                )
                    except (NotImplementedError, TypeError, AttributeError):
                        continue

        # Physics-based numerical dispersion check (Taflove Ch. 4).
        # Instead of a fixed aspect-ratio heuristic, compute the actual
        # per-axis phase velocity error at freq_max from the FDTD
        # dispersion relation. This is application-independent.
        self._check_numerical_dispersion()

        # Thin-metal-on-NU-mesh symmetry (Meep/OpenEMS convention — issue #48).
        self._validate_thin_metal_on_nu_mesh()

    def _check_numerical_dispersion(self) -> None:
        """Warn when per-axis FDTD phase velocity error at freq_max
        exceeds a threshold (Taflove Ch. 4 dispersion relation).

        For each axis the worst-case phase velocity is:
            v_ph = (omega·dt) / (2·arcsin(nu_i · sin(k·d_i/2)))
        where nu_i = c·dt/d_i, k = 2π/λ, d_i = cell size along axis i.

        Reports the per-axis error so the user sees which axis is under-
        resolved or has Courant mismatch — no arbitrary ratio threshold.
        """
        import warnings as _w

        # Skip host-side min when any profile is a tracer. The dispersion
        # warning is advisory only; mesh-as-design-variable optimisation
        # runs under tracing and the warning cannot fire correctly there.
        if any(
            p is not None and is_tracer(p)
            for p in (self._dx_profile, self._dy_profile, self._dz_profile)
        ):
            return

        dx_nom = self._dx or (C0 / self._freq_max / 20.0)
        d = [dx_nom, dx_nom, dx_nom]
        if self._dx_profile is not None:
            d[0] = float(np.min(self._dx_profile))
        if self._dy_profile is not None:
            d[1] = float(np.min(self._dy_profile))
        if self._dz_profile is not None:
            d[2] = float(np.min(self._dz_profile))

        inv_sq = sum(1.0 / di ** 2 for di in d)
        dt_cfl = 0.99 / (C0 * math.sqrt(inv_sq))
        omega = 2.0 * math.pi * self._freq_max

        errors = {}
        sin_wdt2 = math.sin(omega * dt_cfl / 2.0)
        for ax, (name, di) in enumerate(zip("xyz", d)):
            # Taflove Eq. 4.44: v_ph along axis i
            # = omega * d_i / (2 * arcsin(d_i * sin(omega*dt/2) / (c*dt)))
            arg = di * sin_wdt2 / (C0 * dt_cfl)
            if abs(arg) >= 1.0:
                errors[name] = float("inf")
                continue
            v_ph = omega * di / (2.0 * math.asin(arg))
            errors[name] = abs(v_ph - C0) / C0

        max_err = max(errors.values())
        if max_err > 0.02:
            parts = ", ".join(
                f"{name}={err*100:.1f}%" for name, err in errors.items()
            )
            worst = max(errors, key=errors.get)
            _w.warn(
                PreflightWarning(
                    f"FDTD numerical dispersion at freq_max="
                    f"{self._freq_max/1e9:.2f}GHz exceeds 2%: {parts}. "
                    f"Worst axis: {worst} (cell {d['xyz'.index(worst)]*1e3:.3f}mm). "
                    f"Phase velocity error causes resonance frequency bias. "
                    f"Refine the coarse axis or co-refine all axes together "
                    f"(Taflove Ch. 4).",
                    code="numerical_dispersion",
                    source="_check_numerical_dispersion",
                ),
                stacklevel=4,
            )

    def _validate_thin_metal_on_nu_mesh(self) -> None:
        """Warn when a thin PEC sheet sits on a NU axis without symmetric
        neighbouring cells (Meep/OpenEMS require equal dz on both sides of
        a metal plane, else surface currents pick up O(1) error and the
        far-field pattern is corrupted — issue #48).
        """
        import warnings as _w
        profiles = (
            ("x", self._dx_profile),
            ("y", self._dy_profile),
            ("z", self._dz_profile),
        )
        for axis_name, prof in profiles:
            if prof is None:
                continue
            if is_tracer(prof):
                # Tracer profiles can't be host-scanned for edge / ratio
                # checks. The warning is advisory only; correctness is
                # preserved downstream.
                continue
            prof_arr = np.asarray(prof, dtype=np.float64)
            if len(prof_arr) < 3:
                continue
            axis_idx = "xyz".index(axis_name)
            for entry in self._geometry:
                mat = self._resolve_material(entry.material_name)
                if mat.sigma < self._PEC_SIGMA_THRESHOLD:
                    continue
                try:
                    c1, c2 = entry.shape.bounding_box()
                except Exception:
                    continue
                lo, hi = float(c1[axis_idx]), float(c2[axis_idx])
                extent = hi - lo
                min_d = float(prof_arr.min())
                if extent > min_d * 1.5:
                    continue
                # _dz_profile is the user's interior profile (no CPML
                # padding). Geometry coordinates are in interior space
                # starting at 0, so cumsum gives the cell edges directly.
                edges = np.concatenate([[0.0], np.cumsum(prof_arr)])
                mid = 0.5 * (lo + hi)
                k = int(np.searchsorted(edges, mid) - 1)
                if k < 0 or k + 1 >= len(prof_arr) or k - 1 < 0:
                    continue
                dz_here = prof_arr[k]
                dz_above = prof_arr[k + 1]
                dz_below = prof_arr[k - 1]
                # Check ratio both directions — metal-in-coarse-cell
                # next to a fine region is just as bad as the reverse.
                def _ratio(a, b):
                    return max(a, b) / min(a, b)
                ratio_above = _ratio(dz_above, dz_here)
                ratio_below = _ratio(dz_below, dz_here)
                if max(ratio_above, ratio_below) > 1.5:
                    _w.warn(
                        PreflightWarning(
                            f"Thin PEC '{entry.material_name}' on axis "
                            f"{axis_name} sits in a cell of dz={dz_here*1e3:.3f}"
                            f"mm with asymmetric neighbours "
                            f"(below {dz_below*1e3:.3f}, above "
                            f"{dz_above*1e3:.3f} mm). Meep/OpenEMS require "
                            f"equal cell sizes across a metal plane; "
                            f"radiation pattern may be corrupted (issue #48). "
                            f"Put the metal on a preserved-region boundary "
                            f"or refine the neighbouring cell.",
                            code="thin_metal_nu_mesh",
                            source="_validate_thin_metal_on_nu_mesh",
                        ),
                        stacklevel=4,
                    )

    def _check_waveguide_port_evanescent(self) -> None:
        """Warn when measurement frequencies exceed 0.90 × fc_next for any port.

        At f/fc_next > 0.90 the evanescent decay constant is short enough
        that the next higher mode leaks into the single-mode extractor.
        Empirically (40 mm × 20 mm guide, 74 mm port-short spacing):
          f/fc_next = 0.87 → 0.3 % contamination — acceptable for |S11| gate 0.99
          f/fc_next = 0.93 → 1.5 % contamination — registers as |S11| < 1

        Uses port.freqs (measurement freqs) when set; falls back to freq_max.
        """
        import warnings as _w

        for entry in self._waveguide_ports:
            axis = entry.direction[1]  # 'x', 'y', or 'z'
            if axis == "x":
                dim0 = (entry.y_range[1] - entry.y_range[0]
                        if entry.y_range is not None else self._domain[1])
                dim1 = (entry.z_range[1] - entry.z_range[0]
                        if entry.z_range is not None else self._domain[2])
            elif axis == "y":
                dim0 = (entry.x_range[1] - entry.x_range[0]
                        if entry.x_range is not None else self._domain[0])
                dim1 = (entry.z_range[1] - entry.z_range[0]
                        if entry.z_range is not None else self._domain[2])
            else:
                dim0 = (entry.x_range[1] - entry.x_range[0]
                        if entry.x_range is not None else self._domain[0])
                dim1 = (entry.y_range[1] - entry.y_range[0]
                        if entry.y_range is not None else self._domain[1])

            a, b = max(dim0, dim1), min(dim0, dim1)
            if a <= 0 or b <= 0:
                continue

            m0, n0 = entry.mode

            def _fc(m, n, _a=a, _b=b):
                return (C0 / 2.0) * math.sqrt((m / _a) ** 2 + (n / _b) ** 2)

            fc_excited = _fc(m0, n0)
            fc_next = min(
                (
                    _fc(m, n)
                    for m in range(0, 4)
                    for n in range(0, 4)
                    if not (m == 0 and n == 0)
                    and not (m == m0 and n == n0)
                    and _fc(m, n) > fc_excited * (1 + 1e-6)
                ),
                default=None,
            )
            if fc_next is None:
                continue

            if entry.freqs is not None:
                f_check = float(np.max(np.asarray(entry.freqs)))
            else:
                f_check = self._freq_max

            threshold = 0.90 * fc_next
            if f_check > threshold:
                mn_next = min(
                    ((m, n) for m in range(0, 4) for n in range(0, 4)
                     if not (m == 0 and n == 0) and not (m == m0 and n == n0)
                     and abs(_fc(m, n) - fc_next) < 1.0),
                    default=(None, None),
                )
                next_label = (f"TE{mn_next[0]}{mn_next[1]}"
                              if mn_next[0] is not None else "next")
                _w.warn(
                    PreflightWarning(
                        f"Waveguide port '{entry.name}': max measurement frequency "
                        f"{f_check / 1e9:.3f} GHz exceeds 0.90 × fc_next="
                        f"{threshold / 1e9:.3f} GHz "
                        f"(fc_{entry.mode_type}{m0}{n0}={fc_excited / 1e9:.3f} GHz, "
                        f"fc_{next_label}={fc_next / 1e9:.3f} GHz). "
                        f"Evanescent {next_label} contamination may exceed 1 % and "
                        f"registers as |S11| < 1 in a lossless structure. "
                        f"Restrict measurement freqs below {threshold / 1e9:.3f} GHz "
                        f"or increase port-to-obstacle distance.",
                        code="port_evanescent",
                        source="_check_waveguide_port_evanescent",
                    ),
                    stacklevel=4,
                )

    def preflight(
        self,
        *,
        strict: bool = False,
        check_ntff: bool = True,
        check_resolution: bool = True,
        check_ad_memory: bool = False,
        n_steps_for_memory: int | None = None,
        available_memory_gb: float | None = None,
    ) -> "PreflightReport":
        """Run all pre-simulation checks and return warnings.

        Parameters
        ----------
        strict : bool
            If True, raise ValueError on the first issue instead of
            collecting warnings.
        check_ntff : bool
            Run inverse-design NTFF checks (PEC overlap hard-error,
            λ/4 near-field gap warning). Default True.
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
            Empty if no issues found.
        """
        import warnings
        issues = PreflightReport()

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                if check_resolution:
                    self._validate_mesh_quality()
                self._validate_simulation_config()
                if check_ntff:
                    self._validate_ntff_inverse_design()
            except ValueError as e:
                # Collect (do NOT fail-on-first): the aggregated raise at the
                # end escalates every finding at once under strict.
                # Structurally-impossible configs raise PreflightConfigError
                # with the slug set at the check site; any other ValueError is
                # error-severity but uncoded.
                issues.append(PreflightIssue(
                    f"ERROR: {e}",
                    severity="error",
                    code=getattr(e, "code", "uncoded"),
                    loc=getattr(e, "loc", None),
                    source=getattr(e, "source", None),
                ))

        for w in caught:
            msg = str(w.message)
            # Collect (do NOT fail-on-first): aggregated raise at the end.
            # Prefer the structured fields carried on the warning INSTANCE
            # (PreflightWarning); fall back to the category-derived severity for
            # the legacy ``warnings.warn(msg, PreflightErrorWarning)`` form, and
            # to severity="warning"/code="uncoded" for any plain UserWarning.
            inst = w.message
            if isinstance(inst, PreflightWarning):
                severity = inst.severity
                code = inst.code
                loc = inst.loc
                source = inst.source
            else:
                severity = (
                    "error" if issubclass(w.category, PreflightErrorWarning)
                    else "warning"
                )
                code = "uncoded"
                loc = None
                source = None
            issues.append(PreflightIssue(
                msg, severity=severity, code=code, loc=loc, source=source
            ))

        if check_ad_memory:
            if n_steps_for_memory is None:
                raise ValueError("check_ad_memory=True requires n_steps_for_memory")
            est = self.estimate_ad_memory(
                n_steps_for_memory,
                available_memory_gb=available_memory_gb,
            )
            if est.warning:
                issues.append(PreflightIssue(
                    est.warning, severity="warning", code="ad_memory"
                ))

        if strict and issues:
            # Aggregate-then-raise: escalate ALL findings at once. Preserves the
            # historical "strict escalates any issue to ValueError" contract,
            # but reports every problem in one pass instead of fail-on-first
            # (pydantic / Tidy3D pattern). For an errors-only gate that lets
            # advisories through, call ``report.raise_for_failure()`` on a
            # ``strict=False`` report instead.
            raise ValueError(
                f"preflight (strict) found {len(issues)} issue(s):\n  - "
                + "\n  - ".join(issues)
            )

        if issues:
            for iss in issues:
                print(f"  [PREFLIGHT] {iss}")
        else:
            print("  [PREFLIGHT] All checks passed.")

        return issues

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
          uniform single-device S11 vectors.
        - ``calculator="msl"`` checks ``compute_msl_s_matrix(...)``.
        - ``calculator="waveguide"`` checks ``compute_waveguide_s_matrix(...)``.

        Parameters
        ----------
        calculator:
            One of ``"run"``, ``"forward"``, ``"msl"``, or ``"waveguide"``
            (the corresponding method names are accepted as aliases).
        strict:
            If True, raise the underlying ``ValueError`` /
            ``NotImplementedError`` instead of returning it as an issue string.
        normalize:
            Waveguide non-uniform preflight uses this to mirror
            ``compute_waveguide_s_matrix(normalize=...)``.  ``None`` means the
            method default, currently ``False``.
        include_general:
            If True, append the ordinary geometry/material ``preflight()``
            issues after the S-parameter routing check.

        Returns
        -------
        list of str
            Empty when the selected calculator is valid for the registered
            port families.  Otherwise contains actionable issue messages.
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
            "compute_coaxial_s_matrix": "coaxial",
        }
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
            # Collect as a coded error-severity issue (aggregated raise below
            # under strict) — consistent with preflight()'s PreflightIssue
            # contract instead of the old bare f-string.
            issues.append(PreflightIssue(
                f"{type(exc).__name__}: {exc}",
                severity="error",
                code=getattr(exc, "code", f"sparam_routing_{key}"),
                source="preflight_sparameters",
            ))

        if include_general:
            # strict=False here: collect the general findings, then aggregate
            # everything in one raise below (don't fail-on-first).
            issues.extend(self.preflight(strict=False))

        if strict and issues:
            raise ValueError(
                f"preflight_sparameters (strict) found {len(issues)} issue(s):"
                "\n  - " + "\n  - ".join(issues)
            )

        if issues:
            for issue in issues:
                print(f"  [SPARAM PREFLIGHT] {issue}")
        else:
            print(f"  [SPARAM PREFLIGHT] {key}: all checks passed.")
        return issues

    def _validate_msl_sparameter_request_for_preflight(self) -> None:
        """Mirror ``compute_msl_s_matrix`` family-routing checks."""

        if not self._msl_ports:
            raise ValueError("No MSL ports registered. Call add_msl_port() first.")
        if self._ports or self._waveguide_ports or self._floquet_ports:
            raise NotImplementedError(
                "compute_msl_s_matrix() is defined only for add_msl_port(...) "
                "families in the current simulation. Use separate simulations "
                "for add_port(...), add_waveguide_port(...), or "
                "add_floquet_port(...) S-parameter workflows."
            )
        if self._tfsf is not None:
            raise NotImplementedError(
                "compute_msl_s_matrix() is not supported together with TFSF; "
                "TFSF is a plane-wave source, not an MSL port."
            )
        if self._coaxial_ports:
            raise NotImplementedError(
                "compute_msl_s_matrix() does not include add_coaxial_port(...); "
                "coaxial-port S-parameters need a separate validated V/I "
                "extraction and calibration contract."
            )
        if (
            self._dz_profile is not None
            or self._dx_profile is not None
            or self._dy_profile is not None
        ):
            raise NotImplementedError(
                "compute_msl_s_matrix() currently supports the uniform Yee "
                "lane only. Drop dx_profile/dy_profile/dz_profile or use a "
                "documented diagnostic path."
            )
        if self._refinement is not None:
            raise NotImplementedError(
                "compute_msl_s_matrix() is not supported with SBP-SAT "
                "subgridding."
            )
        if self._solver == "adi":
            raise NotImplementedError(
                "compute_msl_s_matrix() is not supported with solver='adi'; "
                "use the uniform Yee solver."
            )

    def _validate_waveguide_sparameter_request_for_preflight(
        self,
        *,
        normalize: bool | str,
    ) -> None:
        """Mirror ``compute_waveguide_s_matrix`` family-routing checks."""

        if not self._waveguide_ports:
            raise ValueError(
                "No waveguide ports registered. Call add_waveguide_port() first."
            )
        if self._ports or self._tfsf:
            raise ValueError(
                "compute_waveguide_s_matrix() is not supported together with "
                "lumped ports or TFSF"
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
        if self._dx_profile is not None or self._dy_profile is not None:
            unsupported = []
            if normalize is not True:
                unsupported.append("normalize=True is required")
            if any(entry.n_modes > 1 for entry in entries):
                unsupported.append("multi-mode ports (n_modes>1) are not supported")
            if unsupported:
                raise NotImplementedError(
                    "compute_waveguide_s_matrix() on a non-uniform mesh "
                    "(dx_profile / dy_profile) supports normalize=True "
                    "and single-mode ports. "
                    + "; ".join(unsupported)
                    + ". Drop the dx/dy profile to use the uniform lane."
                )

    def _validate_coaxial_sparameter_request_for_preflight(self) -> None:
        """Mirror ``compute_coaxial_s_matrix`` family-routing checks."""

        if not self._coaxial_ports:
            raise ValueError(
                "No coaxial ports registered. Call add_coaxial_port() first."
            )
        if (
            self._ports
            or self._waveguide_ports
            or self._floquet_ports
            or self._msl_ports
        ):
            raise NotImplementedError(
                "compute_coaxial_s_matrix() is defined only for "
                "add_coaxial_port(...) families in the current simulation."
            )
        if self._tfsf is not None:
            raise NotImplementedError(
                "compute_coaxial_s_matrix() is not supported together with "
                "TFSF; TFSF is a plane-wave source, not a coaxial port."
            )
        if (
            self._dz_profile is not None
            or self._dx_profile is not None
            or self._dy_profile is not None
        ):
            raise NotImplementedError(
                "compute_coaxial_s_matrix() supports the uniform Yee lane only."
            )
        if self._refinement is not None:
            raise NotImplementedError(
                "compute_coaxial_s_matrix() is not supported with SBP-SAT subgridding."
            )
        if self._solver == "adi":
            raise NotImplementedError(
                "compute_coaxial_s_matrix() is not supported with solver='adi'."
            )

    def _validate_ntff_inverse_design(self) -> None:
        """NTFF inverse-design checks: PEC overlap (error) and λ/4 gap (warn).

        CHECK 2: NTFF face plane strictly intersecting a PEC bbox.
        CHECK 3: NTFF face closer than λ/4 to any geometry/source/probe.
        """
        import warnings as _w

        if self._ntff is None:
            return

        corner_lo, corner_hi, freqs = self._ntff
        # face = (axis, sign, coord, tangential bbox: [(lo_a, hi_a), (lo_b, hi_b)])
        faces = []
        for axis in range(3):
            other = [a for a in range(3) if a != axis]
            tang = ((corner_lo[other[0]], corner_hi[other[0]]),
                    (corner_lo[other[1]], corner_hi[other[1]]))
            faces.append(("lo", axis, corner_lo[axis], tang))
            faces.append(("hi", axis, corner_hi[axis], tang))

        # CHECK 2: strict PEC intersection
        pec_entries = [e for e in self._geometry if e.material_name == "pec"]
        for side, axis, coord, tang in faces:
            for entry in pec_entries:
                try:
                    c1, c2 = entry.shape.bounding_box()
                except (NotImplementedError, TypeError, AttributeError):
                    continue
                # Strict interior along normal axis
                if not (c1[axis] < coord < c2[axis]):
                    continue
                # Tangential overlap along the other two axes
                other = [a for a in range(3) if a != axis]
                overlap = True
                for idx, (tlo, thi) in zip(other, tang):
                    if c2[idx] <= tlo or c1[idx] >= thi:
                        overlap = False
                        break
                if overlap:
                    raise PreflightConfigError(
                        f"NTFF face {'xyz'[axis]}_{side} at {coord*1e3:.2f}mm "
                        f"intersects PEC geometry '{entry.material_name}' "
                        f"(bbox {c1}–{c2}). NTFF box must enclose all radiators "
                        f"with no PEC crossing any face. Shrink or move the NTFF box.",
                        code="ntff_pec_overlap",
                        source="_validate_ntff_inverse_design",
                    )

        # CHECK 3: λ/2 (Huygens) and λ/4 (reactive-near-field) gaps to any
        # geometry/source/probe. Issue #77: the λ/2 Huygens-equivalence rule
        # was documented in papers/rfx-tap/CLAUDE.md but only the λ/4 strong
        # tier was enforced; a face at λ/30 above a ground-plane PEC silently
        # ran and produced corrupted directivity. The two-tier check below
        # warns mildly in [λ/4, λ/2) (results may degrade) and strongly in
        # < λ/4 (directivity / pattern likely corrupted).
        if freqs is None:
            return
        try:
            f_max = float(jnp.max(jnp.asarray(freqs)))
        except Exception:
            f_max = float(self._freq_max)
        lam_min = C0 / max(f_max, 1.0)
        gap_thresh = lam_min / 4.0
        huygens_thresh = lam_min / 2.0

        # Collect candidate bboxes and point positions
        bboxes: list[tuple[str, tuple, tuple]] = []
        for entry in self._geometry:
            try:
                c1, c2 = entry.shape.bounding_box()
                bboxes.append((entry.material_name, c1, c2))
            except (NotImplementedError, TypeError, AttributeError):
                continue
        points: list[tuple[str, tuple]] = []
        for pe in self._ports:
            points.append(("port/source", tuple(pe.position)))
        for pe in self._probes:
            points.append(("probe", tuple(pe.position)))

        for side, axis, coord, tang in faces:
            other = [a for a in range(3) if a != axis]
            min_gap = float("inf")
            culprit = None
            # bbox distances
            for name, c1, c2 in bboxes:
                # tangential overlap check — only meaningful gap if the face
                # is "above" the feature in the normal direction
                overlap = True
                for idx, (tlo, thi) in zip(other, tang):
                    if c2[idx] <= tlo or c1[idx] >= thi:
                        overlap = False
                        break
                if not overlap:
                    continue
                if coord <= c1[axis]:
                    d = c1[axis] - coord
                elif coord >= c2[axis]:
                    d = coord - c2[axis]
                else:
                    d = 0.0  # already handled by CHECK 2 for PEC; skip
                    continue
                if d < min_gap:
                    min_gap, culprit = d, f"geometry '{name}'"
            # points
            for name, pos in points:
                # require tangential in-box for relevance
                in_tang = all(
                    tang[i][0] <= pos[other[i]] <= tang[i][1] for i in range(2)
                )
                if not in_tang:
                    continue
                d = abs(coord - pos[axis])
                if d < min_gap:
                    min_gap, culprit = d, f"{name} at {pos}"

            if culprit is not None and min_gap < gap_thresh:
                _w.warn(
                    PreflightWarning(
                        f"NTFF face {'xyz'[axis]}_{side} is {min_gap*1e3:.2f}mm "
                        f"from {culprit} — below λ/4 = {gap_thresh*1e3:.2f}mm at "
                        f"f_max={f_max/1e9:.2f}GHz. NTFF will integrate reactive "
                        f"near-field; directivity / pattern likely corrupted. "
                        f"Move NTFF box ≥ λ/2 from any radiating/scattering "
                        f"structure (Huygens-equivalence rule).",
                        code="ntff_near_field",
                        source="_validate_ntff_inverse_design",
                    ),
                    stacklevel=3,
                )
            elif culprit is not None and min_gap < huygens_thresh:
                _w.warn(
                    PreflightWarning(
                        f"NTFF face {'xyz'[axis]}_{side} is {min_gap*1e3:.2f}mm "
                        f"from {culprit} — below λ/2 = {huygens_thresh*1e3:.2f}mm "
                        f"at f_max={f_max/1e9:.2f}GHz. Close to reactive near-"
                        f"field; far-field pattern accuracy may degrade. Move "
                        f"NTFF box ≥ λ/2 from radiating/scattering structures.",
                        code="ntff_near_field",
                        source="_validate_ntff_inverse_design",
                    ),
                    stacklevel=3,
                )

    def _validate_simulation_config(self) -> None:
        """Comprehensive pre-simulation configuration validation.

        Checks for common setup mistakes that produce silent wrong results:
        probe/source in CPML, boundary type mismatch, feature compatibility,
        NTFF precision, normalize defaults.

        Called from run() after _validate_mesh_quality().

        Stage 1b refactor (2026-05-17): the original ~592-line body was
        decomposed into per-check ``_validate_cfg_*`` helpers. This method
        keeps its signature and remains the public entry point; its body
        computes the shared local state (``dx``, CPML thicknesses,
        ``absorber_label``) and then calls each helper IN THE SAME ORDER
        as the original checks. No logic, ordering, or warning text
        changed — pure readability decomposition.
        """
        import warnings as _w

        dx = self._dx or C0 / self._freq_max / 20.0
        cpml_thickness = self._cpml_layers * dx if self._boundary in ("cpml", "upml") else 0

        cpml_thick_lo, cpml_thick_hi, _pmc_faces_set = (
            self._validate_cfg_compute_cpml_thickness(cpml_thickness)
        )
        absorber_label = "UPML" if self._boundary == "upml" else "CPML"

        # --- checks in original order ---------------------------------
        self._validate_cfg_pec_faces_with_finite_pec(_w)
        self._validate_cfg_upml_refinement()
        self._validate_cfg_floquet_nonuniform()
        self._validate_cfg_absorber_placement(
            _w, cpml_thickness, cpml_thick_lo, cpml_thick_hi, absorber_label
        )
        self._validate_cfg_source_on_reflector_plane(_w, dx, _pmc_faces_set)
        self._validate_cfg_ntff_absorber_overlap(
            _w, cpml_thickness, cpml_thick_lo, cpml_thick_hi, absorber_label
        )
        self._validate_cfg_ntff_min_steps(dx)
        self._validate_cfg_geometry_in_cpml(
            _w, dx, cpml_thickness, cpml_thick_lo, cpml_thick_hi, absorber_label
        )
        self._validate_cfg_port_inside_pec(_w, dx)
        self._validate_cfg_floating_single_cell_port(_w)
        self._validate_cfg_pec_boundary_open_structure(_w)
        self._validate_cfg_no_sources(_w)
        self._validate_cfg_nonuniform_limitations(_w, cpml_thickness)
        self._validate_cfg_subgrid_limitations(_w)
        self._validate_cfg_conformal_fine_dx(dx)
        self._validate_cfg_lossless_resonator_in_absorber(_w)
        self._validate_cfg_waveguide_reference_plane(
            _w, cpml_thick_lo, cpml_thick_hi
        )

        self._check_waveguide_port_evanescent()
        self._check_msl_port_geometry(dx, cpml_thick_lo, cpml_thick_hi)

    def _validate_cfg_conformal_fine_dx(self, dx: float) -> None:
        """Flag the KNOWN conformal-PEC fine-mesh NaN before it wastes a run.

        ``Boundary(conformal=True)`` / conformal faces at a min cell size
        <= ~2 mm drives the field to NaN. Root cause is a discrete-adjointness
        break (the E-update-only ``eps_eff=eps/w`` makes the update operator
        non-SPSD), NOT a CFL issue — reducing dt does not cure it, and four fix
        methods are falsified (see docs/agent-memory/rfx-known-issues.md, the
        conformal-PEC item). ``normalize=False`` is NOT a safe workaround.
        Surfacing this at preflight converts a silent-NaN GPU run into an
        instant redirect. Emitted as :class:`PreflightErrorWarning` (error
        severity) so an agent can gate on it without it masking other checks.
        """
        spec = getattr(self, "_boundary_spec", None)
        if spec is None or not hasattr(spec, "conformal_faces"):
            return
        try:
            cf = spec.conformal_faces()
        except Exception:
            return
        if not cf:
            return
        cells = [
            c for c in (
                self._dx,
                getattr(self, "_dy", None),
                getattr(self, "_dz", None),
            )
            if c
        ]
        min_cell = min(cells) if cells else dx
        if min_cell <= 2.0e-3:
            import warnings as _w
            # WARNING severity (NOT error/forbid): the conformal-fine-dx NaN is
            # a KNOWN, actively-worked bug, and convergence/development tests
            # must still be able to RUN this config — a hard-fail would block
            # the very work fixing it. Agents gate on the code (conformal_nan),
            # not on a hard-stop.
            # MAINTENANCE: delete this guard when the BCK/USC contour-FIT
            # redesign lands. The strict-xfail tracker
            # tests/test_subpixel_pec.py::test_mesh_convergence_s21_with_conformal_pec
            # will hard-fail (XPASS) to force this removal. See
            # docs/agent-memory/rfx-known-issues.md (conformal-PEC item).
            _w.warn(
                PreflightWarning(
                    f"conformal PEC is enabled on faces {sorted(cf)} with a min "
                    f"cell size {min_cell * 1e3:.3f} mm <= 2 mm — this is a KNOWN "
                    f"NaN (discrete-adjointness break, not CFL; 4 fix methods "
                    f"falsified — see docs/agent-memory/rfx-known-issues.md). "
                    f"normalize=False is NOT a safe workaround at fine dx. Use "
                    f"conformal=False (staircase PEC) or a coarser mesh "
                    f"(dx > 2 mm) until the contour-FIT redesign lands.",
                    code="conformal_nan",
                    severity="warning",
                    source="_validate_cfg_conformal_fine_dx",
                ),
                stacklevel=2,
            )

    def _validate_cfg_lossless_resonator_in_absorber(self, _w) -> None:
        """Warn when EVERY dielectric is perfectly lossless in an open (CPML/
        UPML) domain — design-guide Anti-Pattern #1: a lossless substrate in an
        open boundary yields an artificially infinite Q that reads as a
        plausible-but-wrong resonance (an R5 surface-metric trap detectable
        purely from setup). Deliberately narrow + single-shot to avoid noise:
        it fires only when no dielectric carries any loss, and is hedged
        because it is harmless if you are not measuring Q.
        """
        if self._boundary not in ("cpml", "upml"):
            return
        try:
            from rfx.api._spec import MATERIAL_LIBRARY
        except Exception:
            MATERIAL_LIBRARY = {}

        def _resolve(name):
            mspec = self._materials.get(name) if self._materials else None
            if mspec is not None:
                return (
                    float(getattr(mspec, "eps_r", 1.0)),
                    float(getattr(mspec, "sigma", 0.0)),
                    bool(getattr(mspec, "debye_poles", None)
                         or getattr(mspec, "lorentz_poles", None)),
                )
            lib = MATERIAL_LIBRARY.get(name)
            if isinstance(lib, dict):
                return (
                    float(lib.get("eps_r", 1.0)),
                    float(lib.get("sigma", 0.0)),
                    bool(lib.get("debye_poles") or lib.get("lorentz_poles")),
                )
            return None

        lossless_names: list[str] = []
        any_lossy_dielectric = False
        for entry in self._geometry:
            resolved = _resolve(entry.material_name)
            if resolved is None:
                continue
            eps_r, sigma, has_poles = resolved
            # Dielectric (not vacuum/air, not a conductor/PEC).
            if not (eps_r > 1.05 and sigma < 1.0):
                continue
            if sigma <= 0.0 and not has_poles:
                lossless_names.append(entry.material_name)
            else:
                any_lossy_dielectric = True

        if lossless_names and not any_lossy_dielectric:
            uniq = sorted(set(lossless_names))
            _w.warn(
                PreflightWarning(
                    f"all dielectric(s) {uniq} are perfectly lossless in an open "
                    f"({self._boundary.upper()}) domain. If you are measuring "
                    f"Q / resonance, this gives an ARTIFICIALLY infinite Q "
                    f"(design-guide Anti-Pattern #1, an R5 surface-metric trap) — "
                    f"add loss, e.g. sigma = 2*pi*f*eps0*eps_r*tan_delta. "
                    f"(Harmless if you are not measuring Q.)",
                    code="lossless_q",
                    source="_validate_cfg_lossless_resonator_in_absorber",
                ),
                stacklevel=2,
            )

    def _validate_cfg_compute_cpml_thickness(
        self, cpml_thickness: float
    ) -> tuple[list[float], list[float], set]:
        """Per-face CPML thickness (v1.7.5). Mirrors Grid._face_pad:
        pec_faces / pmc_faces / periodic-axis faces consume 0 cells;
        remaining faces get the axis CPML thickness (non-uniform z
        aggregates the leading dz_profile entries). Under asymmetric
        composition (half-symmetric PMC + CPML, one-sided reflector)
        the lo and hi sides of a single axis can differ — the legacy
        symmetric scalar forced both sides to the max and produced
        false positives on the reflector face.

        Returns ``(cpml_thick_lo, cpml_thick_hi, _pmc_faces_set)``.
        """
        _pmc_faces_set = set(self._boundary_spec.pmc_faces())
        _axis_thickness = [cpml_thickness, cpml_thickness, cpml_thickness]
        if (self._dz_profile is not None
                and not is_tracer(self._dz_profile)
                and self._boundary in ("cpml", "upml")
                and self._cpml_layers > 0):
            n = min(self._cpml_layers, len(self._dz_profile))
            _axis_thickness[2] = float(sum(self._dz_profile[:n]))

        def _face_thickness(ax_idx: int, side: str) -> float:
            ax_name = "xyz"[ax_idx]
            face = f"{ax_name}_{side}"
            if self._boundary not in ("cpml", "upml"):
                return 0.0
            if face in self._pec_faces or face in _pmc_faces_set:
                return 0.0
            if ax_name in self._periodic_axes:
                return 0.0
            return _axis_thickness[ax_idx]

        cpml_thick_lo = [_face_thickness(ax, "lo") for ax in range(3)]
        cpml_thick_hi = [_face_thickness(ax, "hi") for ax in range(3)]
        return cpml_thick_lo, cpml_thick_hi, _pmc_faces_set

    def _validate_cfg_pec_faces_with_finite_pec(self, _w) -> None:
        """Warn about pec_faces + finite PEC objects co-existing.

        pec_faces creates an INFINITE PEC boundary face across the whole
        domain side. Users building antennas or finite-GP structures
        often use pec_faces thinking it's a "ground plane" — but it's
        a full-domain boundary condition, not a finite structure.
        """
        if self._pec_faces and self._geometry:
            has_finite_pec = any(
                entry.material_name == "pec"
                for entry in self._geometry
            )
            if has_finite_pec:
                pec_face_list = ", ".join(sorted(self._pec_faces))
                _w.warn(
                    PreflightWarning(
                        f"pec_faces={{{pec_face_list}}} creates an INFINITE PEC "
                        f"boundary AND the geometry contains finite PEC objects. "
                        f"For antennas or finite-GP structures, the pec_faces "
                        f"boundary makes the ground plane cover the entire domain "
                        f"face, which changes the physics (cavity vs radiating "
                        f"antenna). If you need a finite ground plane, remove "
                        f"pec_faces and use an explicit PEC Box instead.",
                        code="pec_faces_finite_pec",
                        source="_validate_cfg_pec_faces_with_finite_pec",
                    ),
                    stacklevel=3,
                )

    def _validate_cfg_upml_refinement(self) -> None:
        """UPML boundary does not support subgridding/refinement."""
        if self._boundary == "upml" and self._refinement is not None:
            raise PreflightConfigError(
                "boundary='upml' does not support subgridding/refinement",
                code="upml_refinement",
                source="_validate_cfg_upml_refinement",
            )

    def _validate_cfg_floquet_nonuniform(self) -> None:
        """P1.1: Floquet + non-uniform mesh — no silent fallback allowed."""
        if self._floquet_ports and self._dz_profile is not None:
            raise PreflightConfigError(
                "Floquet ports do not support non-uniform z mesh (dz_profile). "
                "Use the uniform reference lane and set dx explicitly.",
                code="floquet_nonuniform",
                source="_validate_cfg_floquet_nonuniform",
            )

    def _validate_cfg_absorber_placement(
        self,
        _w,
        cpml_thickness: float,
        cpml_thick_lo: list[float],
        cpml_thick_hi: list[float],
        absorber_label: str,
    ) -> None:
        """P1.2/P1.3: Probe or source inside absorber region."""
        if cpml_thickness > 0:
            for pe in self._probes:
                pos = pe.position
                for ax, coord in enumerate(pos):
                    domain_extent = self._domain[ax] if ax < len(self._domain) else self._domain[-1]
                    ax_i = min(ax, 2)
                    ct_lo = cpml_thick_lo[ax_i]
                    ct_hi = cpml_thick_hi[ax_i]
                    if coord < ct_lo * 0.5 or coord > domain_extent - ct_hi * 0.5:
                        _w.warn(
                            PreflightWarning(
                                f"Probe at {pos} is near/inside {absorber_label} region "
                                f"({absorber_label} {'xyz'[ax]}-thickness: "
                                f"lo={ct_lo*1e3:.1f}mm, hi={ct_hi*1e3:.1f}mm). "
                                f"Signal will be attenuated. Move probe to interior.",
                                code="absorber_overlap",
                                source="_validate_cfg_absorber_placement",
                            ),
                            stacklevel=3,
                        )
                        break

            for pe in self._ports:
                pos = pe.position
                for ax, coord in enumerate(pos):
                    domain_extent = self._domain[ax] if ax < len(self._domain) else self._domain[-1]
                    ax_i = min(ax, 2)
                    ct_lo = cpml_thick_lo[ax_i]
                    ct_hi = cpml_thick_hi[ax_i]
                    if coord < ct_lo * 0.5 or coord > domain_extent - ct_hi * 0.5:
                        _w.warn(
                            PreflightWarning(
                                f"Source/port at {pos} is near/inside {absorber_label} region "
                                f"({absorber_label} {'xyz'[ax]}-thickness: "
                                f"lo={ct_lo*1e3:.1f}mm, hi={ct_hi*1e3:.1f}mm). "
                                f"Energy will be absorbed. Move source to interior.",
                                code="absorber_overlap",
                                source="_validate_cfg_absorber_placement",
                            ),
                            stacklevel=3,
                        )
                        break

    def _validate_cfg_source_on_reflector_plane(
        self, _w, dx: float, _pmc_faces_set: set
    ) -> None:
        """P1.6: Source / port placed ON a PEC or PMC face plane. Both
        reflectors zero specific field components at the plane every
        time step (PEC: tangential E; PMC: tangential H); a source
        that drives a zeroed component is silently discarded. A
        source that drives a component forced to zero by the mirror
        image (e.g. normal E on a PMC face) fights the symmetry and
        yields numerically inconsistent results.

        Component-specific rule:
          PEC face (axis = ax_name): tangential E (Ex/Ey/Ez with
            component axis != ax_name) is zeroed every E update.
            Normal E (component axis == ax_name) is the legitimate
            way to drive a PEC mirror.
          PMC face (axis = ax_name): tangential H (Hx/Hy/Hz with
            component axis != ax_name) is zeroed; the outgoing
            wave from an on-plane tangential E source is killed via
            this H zeroing. Normal E (component axis == ax_name) is
            odd-symmetric and must be zero at the plane by image,
            so injecting it fights the mirror.

        See docs/research_notes/2026-04-20_source_on_symmetry_plane_industry_survey.md
        for the industry survey behind this rule (Meep / OpenEMS /
        Tidy3D all follow the same convention).
        """
        _all_reflector_faces = set(self._pec_faces) | set(_pmc_faces_set)
        if _all_reflector_faces:
            _dx_axis = [float(dx), float(dx), float(dx)]
            if (self._dz_profile is not None
                    and not is_tracer(self._dz_profile)):
                _dx_axis[2] = float(self._dz_profile[0])
            for face in _all_reflector_faces:
                ax_name = face[0]
                side = face[2:]
                ax_i = "xyz".index(ax_name)
                face_kind = "PMC" if face in _pmc_faces_set else "PEC"
                d_ext = self._domain[ax_i] if ax_i < len(self._domain) else self._domain[-1]
                plane_coord = 0.0 if side == "lo" else float(d_ext)
                tol = 0.5 * _dx_axis[ax_i]
                for pe in self._ports:
                    pos = pe.position
                    coord = pos[ax_i]
                    if abs(coord - plane_coord) > tol:
                        continue
                    # Classify the source component vs. the face axis.
                    comp = pe.component.lower()
                    comp_field = comp[0]       # 'e' or 'h'
                    comp_axis = comp[1:]       # 'x' / 'y' / 'z'
                    is_tangential = (comp_axis != ax_name)
                    if face_kind == "PMC":
                        if comp_field == "e" and is_tangential:
                            msg = (
                                f"Source/port at {pos} (component={pe.component}) "
                                f"sits on the PMC {face} plane. The outgoing "
                                f"tangential H is zeroed every step by "
                                f"apply_pmc_faces, so no wave radiates — the "
                                f"probe records silent zero field. Offset by "
                                f"one cell ({_dx_axis[ax_i]*1e3:.3g} mm) off "
                                f"the plane to let the Yee curl run normally."
                            )
                        elif comp_field == "e" and not is_tangential:
                            msg = (
                                f"Source/port at {pos} (component={pe.component}) "
                                f"sits on the PMC {face} plane and drives the "
                                f"NORMAL E component. PMC imposes odd symmetry "
                                f"on normal E (it must be zero at the plane), "
                                f"so the source fights the mirror image. Use a "
                                f"tangential E source offset by one cell "
                                f"({_dx_axis[ax_i]*1e3:.3g} mm) off the plane."
                            )
                        elif comp_field == "h" and is_tangential:
                            msg = (
                                f"Source/port at {pos} (component={pe.component}) "
                                f"sits on the PMC {face} plane and drives a "
                                f"tangential H. apply_pmc_faces zeros this "
                                f"component at the plane every step, so the "
                                f"source has no effect."
                            )
                        else:
                            msg = None      # normal H on PMC plane is legit
                    else:                    # PEC
                        if comp_field == "e" and is_tangential:
                            msg = (
                                f"Source/port at {pos} (component={pe.component}) "
                                f"sits on the PEC {face} plane and drives a "
                                f"tangential E. PEC zeros E_tan at the plane "
                                f"every step, so the source is silently "
                                f"discarded. Use a normal E source at this "
                                f"face, or offset by one cell "
                                f"({_dx_axis[ax_i]*1e3:.3g} mm) off the plane."
                            )
                        elif comp_field == "h" and not is_tangential:
                            msg = (
                                f"Source/port at {pos} (component={pe.component}) "
                                f"sits on the PEC {face} plane and drives the "
                                f"NORMAL H component. PEC imposes odd symmetry "
                                f"on normal H (it must be zero at the plane). "
                                f"Use a tangential H source or offset by one "
                                f"cell ({_dx_axis[ax_i]*1e3:.3g} mm) off the plane."
                            )
                        else:
                            msg = None      # tangential H or normal E on PEC is legit
                    if msg is not None:
                        _w.warn(
                            PreflightWarning(
                                msg,
                                code="source_decoupled",
                                source="_validate_cfg_source_on_reflector_plane",
                            ),
                            stacklevel=3,
                        )

    def _validate_cfg_ntff_absorber_overlap(
        self,
        _w,
        cpml_thickness: float,
        cpml_thick_lo: list[float],
        cpml_thick_hi: list[float],
        absorber_label: str,
    ) -> None:
        """P1.4: NTFF box overlap with absorber."""
        if self._ntff is not None and cpml_thickness > 0:
            corner_lo, corner_hi, _ = self._ntff
            for ax in range(3):
                domain_ext = self._domain[ax] if ax < len(self._domain) else self._domain[-1]
                ax_i = min(ax, 2)
                ct_lo = cpml_thick_lo[ax_i]
                ct_hi = cpml_thick_hi[ax_i]
                if corner_lo[ax] < ct_lo or corner_hi[ax] > domain_ext - ct_hi:
                    _w.warn(
                        PreflightWarning(
                            f"NTFF box extends into {absorber_label} region along "
                            f"{'xyz'[ax]}-axis. Far-field results will be "
                            f"corrupted. Shrink NTFF box to interior.",
                            code="absorber_overlap",
                            source="_validate_cfg_ntff_absorber_overlap",
                        ),
                        stacklevel=3,
                    )
                    break

        # P1.5: (merged into P2.1 — non-uniform + NTFF is unsupported)

    def _validate_cfg_ntff_min_steps(self, dx: float) -> None:
        """P1.7: NTFF with too few steps."""
        if self._ntff is not None:
            _, _, ntff_freqs = self._ntff
            if ntff_freqs is not None:
                min_freq = float(min(ntff_freqs))
                period = 1.0 / max(min_freq, 1.0)
                dt_est = dx / (C0 * 1.732) * 0.99  # CFL estimate
                min_steps_for_ntff = int(10 * period / dt_est)
                # Can't check n_steps here (not known yet), but store hint
                self._ntff_min_steps_hint = min_steps_for_ntff

    def _validate_cfg_geometry_in_cpml(
        self,
        _w,
        dx: float,
        cpml_thickness: float,
        cpml_thick_lo: list[float],
        cpml_thick_hi: list[float],
        absorber_label: str,
    ) -> None:
        """P1.9: Geometry (dielectric OR PEC) extending into CPML region.

        CPML modifies field-update equations with absorbing coefficients;
        any structure placed there is effectively eaten by the absorber
        and produces physically meaningless results (issue #61).
        Periodic axes have no CPML (see _build_grid — issue #68), so
        the per-axis thresholds above already carry `cpml_thick_xyz[ax]
        == 0` on those axes and the check naturally skips.
        """
        if cpml_thickness > 0 and self._boundary == "cpml":
            for entry in self._geometry:
                if hasattr(entry.shape, "bounding_box"):
                    try:
                        c1, c2 = entry.shape.bounding_box()
                        for ax in range(min(3, len(self._domain))):
                            thick_lo = cpml_thick_lo[ax]
                            thick_hi = cpml_thick_hi[ax]
                            if thick_lo <= 0 and thick_hi <= 0:
                                continue
                            d = self._domain[ax] if ax < len(self._domain) else self._domain[-1]
                            # FP3 refinement (2026-05-06): a Box whose
                            # lo/hi edge sits exactly at 0/L_domain
                            # (within ½·dx) is an *intentional*
                            # full-domain extension — the canonical
                            # transmission-line / MSL-substrate
                            # pattern.  Do not warn on those edges;
                            # only warn on edges that drift INTO the
                            # CPML region without explicitly reaching
                            # the boundary (the original issue #61
                            # leak-into-absorber footgun).
                            # 2026-06 fix: "touches the edge" alone is not
                            # enough — a thin slab buried ENTIRELY inside the
                            # absorber also starts at the edge. An edge-touching
                            # box is an intentional full-domain extension only if
                            # it reaches PAST the CPML into the physical interior
                            # (c2 > thick_lo); a slab contained within the CPML is
                            # the issue-#61 footgun and must warn (regression:
                            # test_preflight_still_warns_on_non_periodic_z_axis).
                            # BUT when the CPML spans (nearly) the whole axis there
                            # is no interior to reach and no footgun to flag — the
                            # warning is meaningless, so the edge touch stays
                            # intentional (degenerate full-CPML axis, e.g. the
                            # thin-substrate false-positive test where
                            # layers·dx ≈ L_axis).
                            has_interior = (thick_lo + thick_hi) < d - dx * 0.5
                            intentional_lo = c1[ax] <= dx * 0.5 and (
                                c2[ax] > thick_lo or not has_interior
                            )
                            intentional_hi = c2[ax] >= d - dx * 0.5 and (
                                c1[ax] < d - thick_hi or not has_interior
                            )
                            lo_hit = (thick_lo > 0
                                      and c1[ax] < thick_lo * 0.3
                                      and not intentional_lo)
                            hi_hit = (thick_hi > 0
                                      and c2[ax] > d - thick_hi * 0.3
                                      and not intentional_hi)
                            if lo_hit or hi_hit:
                                _w.warn(
                                    PreflightWarning(
                                        f"Material '{entry.material_name}' extends "
                                        f"into CPML region along {'xyz'[ax]}-axis. "
                                        f"{absorber_label} modifies field updates — "
                                        f"geometry inside the absorber is physically "
                                        f"meaningless (issue #61).",
                                        code="geometry_in_absorber",
                                        source="_validate_cfg_geometry_in_cpml",
                                    ),
                                    stacklevel=3,
                                )
                                break
                    except (NotImplementedError, TypeError):
                        pass

    def _validate_cfg_port_inside_pec(self, _w, dx: float) -> None:
        """P1.8: Port/source/probe inside PEC geometry.

        FP4 refinement (2026-05-06): tangential H is non-zero on a
        PEC surface and well-defined within a thin (≤ 1.5·dx) PEC
        sheet — for example, an MSL diagnostic Hy probe placed at
        z = h_sub + 0.5·dx (the centre of a 1-cell trace) measures
        the trace surface current and must not warn.  Inside a thick
        PEC volume H still decays to zero, so the warning still
        fires there.
        """
        for pe in list(self._ports) + list(self._probes):
            pos = pe.position
            component = (getattr(pe, "component", "") or "").lower()
            is_h_component = component in ("hx", "hy", "hz")
            for entry in self._geometry:
                if entry.material_name != "pec":
                    continue
                if hasattr(entry.shape, "bounding_box"):
                    try:
                        c1, c2 = entry.shape.bounding_box()
                        inside = all(c1[ax] <= pos[ax] <= c2[ax] for ax in range(3))
                        if inside:
                            pec_min_thickness = min(
                                c2[i] - c1[i] for i in range(3)
                            )
                            is_thin_pec = pec_min_thickness <= 1.5 * dx
                            if is_h_component and is_thin_pec:
                                continue
                            _w.warn(
                                PreflightWarning(
                                    f"Port/source at {pos} is inside PEC geometry "
                                    f"'{entry.material_name}'. Field will be zero. "
                                    f"Move source outside PEC.",
                                    code="port_in_pec",
                                    source="_validate_cfg_port_inside_pec",
                                ),
                                stacklevel=3,
                            )
                    except (NotImplementedError, TypeError):
                        pass

    def _validate_cfg_floating_single_cell_port(self, _w) -> None:
        """P1.9: Single-cell port in dielectric with no adjacent PEC pin
        (issue #71). A single-cell LumpedPort placed mid-substrate with
        no conducting pin or microstrip does not couple to patch-antenna
        TM modes — the optimiser reads a nonsense loss from the
        floating Ez source. Recommend extent=<substrate_height> to
        promote to a WirePort spanning ground → patch.
        """
        _PORT_COMP_AXIS = {"ex": 0, "ey": 1, "ez": 2}
        for pe in self._ports:
            # Filter: only true ports (impedance > 0), single-cell
            # (extent is None), actively excited (excite is True).
            # add_source() creates _PortEntry with impedance=0.0 and is
            # intentionally a soft source — not a port footgun.
            if not pe.impedance or pe.impedance <= 0.0:
                continue
            if pe.extent is not None:
                continue
            if not pe.excite:
                continue
            pos = pe.position
            # Find the dielectric geometry enclosing the port cell.
            enclosing_eps_r = None
            enclosing_name = None
            for entry in self._geometry:
                if entry.material_name == "pec":
                    continue
                if not hasattr(entry.shape, "bounding_box"):
                    continue
                try:
                    c1, c2 = entry.shape.bounding_box()
                except (NotImplementedError, TypeError):
                    continue
                inside = all(c1[ax] <= pos[ax] <= c2[ax] for ax in range(3))
                if not inside:
                    continue
                mspec = self._materials.get(entry.material_name)
                if mspec is not None and float(mspec.eps_r) > 1.0 + 1e-3:
                    enclosing_eps_r = float(mspec.eps_r)
                    enclosing_name = entry.material_name
                    break
            if enclosing_eps_r is None:
                continue
            # Check for a PEC geometry one cell away along the port's
            # component axis (coax-style pin or microstrip feed edge).
            # Without such a pin, the port cell cannot drive a vertical
            # current that couples to the patch TM mode.
            comp_axis = _PORT_COMP_AXIS.get(pe.component)
            if comp_axis is None:
                continue
            nudge = float(self._dx or 0.0) * 1.01
            adj_positions = (
                tuple(pos[i] + (nudge if i == comp_axis else 0.0) for i in range(3)),
                tuple(pos[i] - (nudge if i == comp_axis else 0.0) for i in range(3)),
            )
            has_adjacent_pec = False
            for apos in adj_positions:
                for entry in self._geometry:
                    if entry.material_name != "pec":
                        continue
                    if not hasattr(entry.shape, "bounding_box"):
                        continue
                    try:
                        c1, c2 = entry.shape.bounding_box()
                    except (NotImplementedError, TypeError):
                        continue
                    if all(c1[ax] <= apos[ax] <= c2[ax] for ax in range(3)):
                        has_adjacent_pec = True
                        break
                if has_adjacent_pec:
                    break
            if has_adjacent_pec:
                continue
            _w.warn(
                PreflightWarning(
                    f"Single-cell port at {pos} ({pe.component}) sits inside "
                    f"dielectric '{enclosing_name}' (eps_r={enclosing_eps_r:.2f}) "
                    f"with no adjacent PEC along the {pe.component[1]}-axis. A "
                    f"floating single-cell port inside substrate does not "
                    f"couple to patch-antenna TM modes. Pass "
                    f"extent=<substrate_height> to create a WirePort spanning "
                    f"ground → patch plane (issue #71).",
                    code="floating_port",
                    source="_validate_cfg_floating_single_cell_port",
                ),
                stacklevel=3,
            )

    def _validate_cfg_pec_boundary_open_structure(self, _w) -> None:
        """P0.4: PEC boundary on likely open structure."""
        if self._boundary == "pec" and self._ntff is not None:
            _w.warn(
                PreflightWarning(
                    "PEC boundary with NTFF far-field: PEC reflects all energy "
                    "back into domain. Use boundary='cpml' or boundary='upml' for open structures "
                    "(antennas, scatterers).",
                    code="pec_boundary_open",
                    source="_validate_cfg_pec_boundary_open_structure",
                ),
                stacklevel=3,
            )

    def _validate_cfg_no_sources(self, _w) -> None:
        """P0.5: No sources configured."""
        if (
            not self._ports
            and self._tfsf is None
            and not self._waveguide_ports
            and not self._floquet_ports
            and not self._msl_ports
        ):
            _w.warn(
                PreflightWarning(
                    "No sources, ports, TFSF, or waveguide/Floquet/MSL ports configured. "
                    "Simulation will produce zero fields.",
                    code="no_sources",
                    source="_validate_cfg_no_sources",
                ),
                stacklevel=3,
            )

    def _validate_cfg_nonuniform_limitations(
        self, _w, cpml_thickness: float
    ) -> None:
        """P2: Non-uniform mesh shadow-lane limitations."""
        if self._dz_profile is not None:
            # P2.3: TFSF on nonuniform mesh — narrowed scope.
            # Axis-aligned ±x incidence with angle_deg=0 runs the 1D
            # auxiliary along the uniform x axis and is supported. The
            # z-directed and oblique cases would need a z-nonuniform 1D
            # aux (resp. nonuniform 2D aux) and are deferred.
            if self._tfsf is not None:
                if self._tfsf.direction in ("+z", "-z"):
                    raise PreflightConfigError(
                        "TFSF z-directed incidence is not yet supported on "
                        "nonuniform z mesh. Axis-aligned incidence along x "
                        "(direction='+x' or '-x') is supported.",
                        code="nonuniform_tfsf",
                        source="_validate_cfg_nonuniform_limitations",
                    )
                if abs(self._tfsf.angle_deg) > 0.01:
                    raise PreflightConfigError(
                        "TFSF oblique incidence is not yet supported on "
                        "nonuniform z mesh. Use angle_deg=0.",
                        code="nonuniform_tfsf",
                        source="_validate_cfg_nonuniform_limitations",
                    )

            # P2.6: CPML z-thickness on non-uniform mesh.
            # Skip on tracer profiles — advisory warning only.
            if (self._boundary == "cpml"
                    and self._cpml_layers > 0
                    and not is_tracer(self._dz_profile)):
                cpml_z_thick = sum(float(d) for d in self._dz_profile[:self._cpml_layers])
                if cpml_z_thick < cpml_thickness * 0.3:
                    _w.warn(
                        PreflightWarning(
                            f"CPML z-thickness is {cpml_z_thick*1e3:.1f}mm "
                            f"({self._cpml_layers} cells), much thinner than "
                            f"xy-thickness {cpml_thickness*1e3:.1f}mm. "
                            f"Absorbing performance may be asymmetric. "
                            f"Consider more z cells or fewer CPML layers.",
                            code="nonuniform_cpml_thin",
                            source="_validate_cfg_nonuniform_limitations",
                        ),
                        stacklevel=3,
                    )

    def _validate_cfg_subgrid_limitations(self, _w) -> None:
        """P4: Subgridded path limitations.

        P3 (Distributed path): distributed warnings are emitted at
        run() dispatch time in distributed_v2.py — no preflight check
        here.
        """
        if self._refinement is not None:
            if self._dft_planes:
                _w.warn(
                    PreflightWarning(
                        "DFT plane probes are not supported with SBP-SAT "
                        "subgridding.",
                        code="subgrid_unsupported_feature",
                        source="_validate_cfg_subgrid_limitations",
                    ),
                    stacklevel=3,
                )
            if self._waveguide_ports:
                _w.warn(
                    PreflightWarning(
                        "Waveguide ports are not supported with SBP-SAT "
                        "subgridding.",
                        code="subgrid_unsupported_feature",
                        source="_validate_cfg_subgrid_limitations",
                    ),
                    stacklevel=3,
                )
            if self._floquet_ports:
                _w.warn(
                    PreflightWarning(
                        "Floquet ports are not supported with SBP-SAT subgridding.",
                        code="subgrid_unsupported_feature",
                        source="_validate_cfg_subgrid_limitations",
                    ),
                    stacklevel=3,
                )
            if self._tfsf is not None:
                _w.warn(
                    PreflightWarning(
                        "TFSF source is not supported with SBP-SAT subgridding.",
                        code="subgrid_unsupported_feature",
                        source="_validate_cfg_subgrid_limitations",
                    ),
                    stacklevel=3,
                )
            if self._lumped_rlc:
                _w.warn(
                    PreflightWarning(
                        "Lumped RLC elements are not supported with SBP-SAT "
                        "subgridding.",
                        code="subgrid_unsupported_feature",
                        source="_validate_cfg_subgrid_limitations",
                    ),
                    stacklevel=3,
                )

    def _validate_cfg_waveguide_reference_plane(
        self,
        _w,
        cpml_thick_lo: list[float],
        cpml_thick_hi: list[float],
    ) -> None:
        """P2.8: Waveguide-port reference plane sanity.

        The S-matrix returned by ``compute_waveguide_s_matrix`` is
        evaluated AT the reference plane (either ``entry.reference_plane``
        if user-specified, or the port's ``x_position`` by default after
        2026-04-22). The phase of reported S-params is therefore tied to
        that plane. Physical correctness requires the plane lies inside
        the simulation domain, outside the CPML absorbing region, and
        preferably inside a uniform-cross-section segment of guide so the
        modal decomposition is defined.

        P2.7 (obsolete): PMC / PEC + CPML on the same axis used to emit
        a warning for the architectural offset between the reflector
        plane and the user domain edge. v1.7.5 closed that gap on both
        the uniform (rfx/grid.py) and non-uniform (rfx/nonuniform.py)
        paths via per-face ``pad_{axis}_{lo,hi}`` allocation. The
        warning is retained as a no-op anchor so external references
        ("[P2.7]") don't break and as a reminder that the fix is
        regression-locked via tests/test_silent_drop_warnings.py and
        tests/test_boundary_pmc_hi_faces.py.
        """
        if self._waveguide_ports:
            axis_map = {"x": 0, "y": 1, "z": 2}
            for entry in self._waveguide_ports:
                direction = entry.direction  # e.g., "+x", "-x"
                ax_i = axis_map[direction[-1]]
                domain_ext = self._domain[ax_i]
                ct_lo = cpml_thick_lo[ax_i]
                ct_hi = cpml_thick_hi[ax_i]
                effective = (entry.reference_plane if entry.reference_plane is not None
                             else entry.x_position)
                if effective < 0 or effective > domain_ext:
                    raise PreflightConfigError(
                        f"waveguide_port reference plane = {effective:.4g} m is "
                        f"outside the {direction[-1]}-domain [0, {domain_ext:.4g}] m. "
                        f"Check x_position / reference_plane.",
                        code="waveguide_reference_plane",
                        source="_validate_cfg_waveguide_reference_plane",
                    )
                if effective < ct_lo or effective > domain_ext - ct_hi:
                    _w.warn(
                        PreflightWarning(
                            f"waveguide_port reference plane = {effective*1e3:.3g} mm is "
                            f"inside the CPML absorbing region along the "
                            f"{direction[-1]}-axis (CPML extent: "
                            f"[0, {ct_lo*1e3:.3g}] and "
                            f"[{(domain_ext - ct_hi)*1e3:.3g}, {domain_ext*1e3:.3g}] mm). "
                            f"S-matrix phase will be distorted by CPML stretching. "
                            f"Move x_position / reference_plane to the interior or "
                            f"reduce cpml_layers.",
                            code="waveguide_reference_plane",
                            source="_validate_cfg_waveguide_reference_plane",
                        ),
                        stacklevel=3,
                    )
                # Device overlap warning: check if any geometry box spans
                # the port's x-plane.
                if self._geometry:
                    for g in self._geometry:
                        try:
                            lo, hi = g.bounds
                        except Exception:
                            continue
                        if lo[ax_i] <= effective <= hi[ax_i]:
                            _w.warn(
                                PreflightWarning(
                                    f"waveguide_port reference plane at "
                                    f"{effective*1e3:.3g} mm intersects geometry "
                                    f"'{getattr(g, 'material', '?')}' "
                                    f"(bounds {lo[ax_i]*1e3:.3g}–{hi[ax_i]*1e3:.3g} mm "
                                    f"on {direction[-1]}). Modal decomposition "
                                    f"assumes a uniform cross-section at the port "
                                    f"plane; reported S-params will mix modes. Move "
                                    f"the reference plane into the empty-guide region.",
                                    code="waveguide_reference_plane",
                                    source="_validate_cfg_waveguide_reference_plane",
                                ),
                                stacklevel=3,
                            )
                            break

    def _check_msl_port_geometry(
        self,
        dx: float,
        cpml_thick_lo: list[float],
        cpml_thick_hi: list[float],
    ) -> None:
        """MSL port setup correctness checks (issue: silent Z0 / |S11| bias).

        Microstrip Z0 and |S11| are extremely sensitive to lateral box
        size and substrate resolution. Wrong setup can give 15-30% Z0
        bias or anti-convergent mesh-conv with no error message.
        Catches the common mistakes here so users find them in <1 min
        instead of after a full mesh sweep.

        Three checks per MSL port:

        1. **Lateral clearance** from trace edge to nearest absorbing
           boundary (CPML/PML) or PEC sidewall must be ≥ 2·h_sub.
           Microstrip fringing fields decay as exp(-π·d/h_sub); the
           5%-amplitude tail sits at d ≈ 0.95·h_sub. A ≥ 2·h_sub margin
           keeps Z0 bias under ~5% (verified by fixed-LY mesh-conv
           sweep, 2026-05-04 — see rfx-known-issues.md).

        2. **Substrate resolution** n_z_sub = h_sub/dx ≥ 4 cells.
           Yee staircase at the dielectric interface is O(dx) (not
           O(dx²)) for inhomogeneous ε; <4 cells gives Z0 staircase
           error >5%.

        3. **Port-to-CPML distance** in propagation direction ≥ 2·h_sub.
           Source-side CPML reflection inflates |S11| if the port is
           too close.
        """
        import warnings as _w
        if not self._msl_ports:
            return
        domain = self._domain
        for pe in self._msl_ports:
            x_feed, y_centre, z_lo = pe.position
            w_trace = float(pe.width)
            h_sub = float(pe.height)
            recommended = 2.0 * h_sub

            # ---- 1. Lateral (y) clearance ----
            trace_y_lo = y_centre - w_trace / 2.0
            trace_y_hi = y_centre + w_trace / 2.0
            ly = float(domain[1])
            # Effective absorbing boundary positions on each y side
            y_abs_lo = float(cpml_thick_lo[1])           # CPML extent from y=0
            y_abs_hi = ly - float(cpml_thick_hi[1])      # CPML extent from y=LY
            clearance_lo = trace_y_lo - y_abs_lo
            clearance_hi = y_abs_hi - trace_y_hi
            for side, c in (("−y", clearance_lo), ("+y", clearance_hi)):
                if c < recommended:
                    pct = max(0.0, (1.0 - c / recommended)) * 15.0
                    _w.warn(
                        PreflightWarning(
                            f"MSL port '{pe.name}' (trace W={w_trace*1e6:.0f}µm, "
                            f"h_sub={h_sub*1e6:.0f}µm): lateral clearance to "
                            f"{side} absorbing boundary = {c*1e6:.0f}µm < "
                            f"recommended {recommended*1e6:.0f}µm (= 2·h_sub). "
                            f"Fringing field will be clipped → Z0 may be biased "
                            f"HIGH by ~{pct:.0f}%, mesh-conv may diverge. "
                            f"Increase domain y-extent OR move port further from "
                            f"sidewall.",
                            code="msl_port_geometry",
                            source="_check_msl_port_geometry",
                        ),
                        stacklevel=3,
                    )

            # ---- 2. Substrate cells ----
            n_z_sub = max(1, int(round(h_sub / dx)))
            if n_z_sub < 4:
                _w.warn(
                    PreflightWarning(
                        f"MSL port '{pe.name}': only {n_z_sub} substrate cell(s) "
                        f"in z (h_sub={h_sub*1e6:.0f}µm, dx={dx*1e6:.0f}µm). "
                        f"Yee staircase at dielectric interface is O(dx) — "
                        f"Z0 staircase error >5% expected. Refine to dx ≤ "
                        f"{h_sub*1e6/4:.0f}µm (4+ substrate cells) for "
                        f"<5% Z0 bias.",
                        code="msl_port_geometry",
                        source="_check_msl_port_geometry",
                    ),
                    stacklevel=3,
                )

            # ---- 2b. Substrate-boundary cell alignment for
            # ``pec_occupancy_override`` users.  When h_sub/dx has a
            # fractional part in [0.10, 0.40], the substrate-air
            # interface lands in the lower portion of a Yee cell that
            # ALSO contains the trace at z=h_sub..h_sub+dx; the cell is
            # mixed substrate + PEC.  Hard-PEC ``Box(material="pec")``
            # handles this via subpixel material assembly, but the
            # AD-traceable ``pec_occupancy_override`` path zeros the
            # whole cell and produces unphysical |S21| (verified
            # 2026-05-08, runs #563/#567: |S21|² > 1 across all stub
            # lengths at dx ∈ [75, 82]µm with h_sub=254µm).  Snap dx
            # so h_sub/dx is integer or its fractional part is > 0.6 to
            # stay in a safe alignment window.
            frac = (h_sub / dx) - int(h_sub / dx)
            if 0.10 <= frac <= 0.40:
                # Snap suggestions: nearest integer above and below.
                n_below = int(h_sub / dx)
                n_above = n_below + 1
                dx_low = h_sub / n_above   # frac=0
                dx_high = h_sub / n_below  # frac=0
                _w.warn(
                    PreflightWarning(
                        f"MSL port '{pe.name}': h_sub/dx = "
                        f"{h_sub/dx:.3f} (fractional part {frac:.3f}) lands "
                        f"in the [0.10, 0.40] mixed-cell danger zone. The "
                        f"substrate-air interface bisects the same Yee cell "
                        f"that holds the trace; AD-traceable "
                        f"``pec_occupancy_override`` zeros the whole cell "
                        f"and produces unphysical |S21|² > 1 in this regime. "
                        f"Hard ``Box(material='pec')`` is unaffected. To "
                        f"snap onto a safe alignment, set dx = "
                        f"{dx_low*1e6:.1f}µm (= h_sub/{n_above}) or "
                        f"{dx_high*1e6:.1f}µm (= h_sub/{n_below}).",
                        code="msl_port_geometry",
                        source="_check_msl_port_geometry",
                    ),
                    stacklevel=3,
                )

            # ---- 3. Port-to-CPML distance in x ----
            x_abs_lo = float(cpml_thick_lo[0])
            x_abs_hi = float(domain[0]) - float(cpml_thick_hi[0])
            x_clearance = (
                x_feed - x_abs_lo if pe.direction == "+x"
                else x_abs_hi - x_feed
            )
            if x_clearance < recommended:
                _w.warn(
                    PreflightWarning(
                        f"MSL port '{pe.name}' at x={x_feed*1e3:.2f}mm, "
                        f"direction={pe.direction!r}: distance to nearest "
                        f"x-CPML = {x_clearance*1e6:.0f}µm < recommended "
                        f"{recommended*1e6:.0f}µm (= 2·h_sub). Source-side "
                        f"CPML reflection may inflate |S11|. Move port further "
                        f"from boundary OR increase domain x-extent.",
                        code="msl_port_geometry",
                        source="_check_msl_port_geometry",
                    ),
                    stacklevel=3,
                )

            # ---- 4. Probe-to-reflector distance — standing-wave bias ----
            # The 3-probe Z0 extractor in compute_msl_s_matrix assumes a
            # CLEAN travelling-wave regime at the V1/V2/V3 probe locations.
            # When a strong reflector (PEC stub, open termination, mismatch)
            # sits within ≲ λ_g/4 of the probes, V_i contains substantial
            # standing-wave content and the recovered (α, γ, Z0) get
            # biased — typically reading |S11| ≪ 1 even when physics
            # demands full reflection.  Catches the cv06b-vs-Y2-demo
            # divergence (cv06b's L_LINE=30mm passes; Y2's L_LINE=5mm
            # fails by ~7 dB on |S11|@notch — see
            # `docs/research_notes/20260506_y2_s11_notch_bias_root_cause.md`).
            #
            # Conservative ε_eff_proxy = 5.0 → upper bound on β → lower
            # bound on λ_g → most stringent (smallest) recommended
            # clearance.  For air-only lines this is overly conservative,
            # but the cost of a false-positive warning is low.
            EPS_EFF_PROXY = 5.0
            f_max = float(self._freq_max)
            c0 = 2.998e8
            lambda_g_min = c0 / (f_max * (EPS_EFF_PROXY ** 0.5))
            # Recommended: probe-to-reflector ≥ λ_g/4 at f_max.  At lower
            # frequencies λ_g is larger and the same physical clearance
            # represents fewer cells of standing-wave-free zone — but
            # f_max is the worst case.
            min_probe_clear = 0.25 * lambda_g_min

            # Last 3-probe x-position (V₃, deepest into the line)
            n_off = pe.n_probe_offset if pe.n_probe_offset is not None else 5
            n_sp = pe.n_probe_spacing if pe.n_probe_spacing is not None else 3
            sign = 1.0 if pe.direction == "+x" else -1.0
            x_v3 = x_feed + sign * (n_off + 2 * n_sp) * dx

            # Walk geometry: find PEC Box reflectors between this port's
            # V3 and the FAR end of the line.  Exclude the through-line
            # trace itself — heuristic: the trace is a Box whose
            # x-extent ≥ 80 % of the inter-port distance and whose
            # y-extent equals the trace width.
            x_far = (float(domain[0]) - x_abs_hi) if pe.direction == "+x" else x_abs_lo
            inter_port_extent = abs(x_far - x_feed)
            from rfx.geometry.csg import Box as _Box
            nearest_d = float("inf")
            nearest_label = None
            for ge in getattr(self, "_geometry", []):
                shape = getattr(ge, "shape", None)
                mat = getattr(ge, "material_name", "")
                if not isinstance(shape, _Box) or str(mat).lower() != "pec":
                    continue
                lo, hi = shape.corner_lo, shape.corner_hi
                box_x_lo, box_x_hi = float(lo[0]), float(hi[0])
                box_y_lo, box_y_hi = float(lo[1]), float(hi[1])
                # Skip the through-line trace itself
                box_x_extent = box_x_hi - box_x_lo
                box_y_extent = box_y_hi - box_y_lo
                if (box_x_extent >= 0.8 * inter_port_extent
                        and abs(box_y_extent - w_trace) <= dx):
                    continue
                # Skip ground plane boxes (the box that spans both
                # transversally AND below the substrate; identify by
                # a thin z-extent below the substrate top)
                if box_y_extent >= 0.8 * float(domain[1]):
                    continue
                # Distance from V3 to the nearest edge of this box,
                # measured ALONG the propagation direction
                if sign > 0:
                    if box_x_lo > x_v3:
                        d = box_x_lo - x_v3
                    elif box_x_hi < x_v3:
                        continue   # behind the probe
                    else:
                        d = 0.0
                else:
                    if box_x_hi < x_v3:
                        d = x_v3 - box_x_hi
                    elif box_x_lo > x_v3:
                        continue
                    else:
                        d = 0.0
                if d < nearest_d:
                    nearest_d = d
                    nearest_label = (
                        f"PEC Box at x∈[{box_x_lo*1e3:.2f},{box_x_hi*1e3:.2f}]mm "
                        f"y∈[{box_y_lo*1e3:.2f},{box_y_hi*1e3:.2f}]mm"
                    )

            if nearest_d < min_probe_clear and nearest_label is not None:
                _w.warn(
                    PreflightWarning(
                        f"MSL port '{pe.name}' (direction={pe.direction!r}): "
                        f"3-probe V₃ at x={x_v3*1e3:.2f}mm sits {nearest_d*1e6:.0f}µm "
                        f"from a strong reflector ({nearest_label}); recommended "
                        f"≥ {min_probe_clear*1e6:.0f}µm "
                        f"(= λ_g/4 at f_max with ε_eff_proxy={EPS_EFF_PROXY:.1f}). "
                        f"Standing-wave content at the probes will bias "
                        f"`compute_msl_s_matrix`'s Z₀ extraction and |S11|@notch — "
                        f"physical |S11|→1 at a quarter-wave open stub may read "
                        f"as -5 to -10 dB instead of 0 dB.  Mitigation: "
                        f"extend L_LINE so the line between port and reflector "
                        f"is ≥ λ_g/2, OR bump n_probe_offset on add_msl_port to "
                        f"push V₃ further into a clean travelling-wave region.",
                        code="msl_port_geometry",
                        source="_check_msl_port_geometry",
                    ),
                    stacklevel=3,
                )

    def _validate_adi_configuration(self, materials: MaterialArrays, debye_spec, lorentz_spec) -> None:
        """Validate that the current simulation is compatible with the ADI path."""
        if self._mode not in ("2d_tmz", "3d"):
            raise ValueError("solver='adi' supports mode='3d' or mode='2d_tmz'")
        if self._boundary == "upml":
            raise ValueError("solver='adi' does not support boundary='upml'")
        if self._boundary not in ("pec", "cpml"):
            raise ValueError("solver='adi' supports boundary='pec' or 'cpml'")
        if self._refinement is not None:
            raise ValueError("solver='adi' does not support subgridding yet")
        if self._tfsf is not None:
            raise ValueError("solver='adi' does not support TFSF sources yet")
        if self._waveguide_ports or self._floquet_ports:
            raise ValueError("solver='adi' does not support waveguide or Floquet ports yet")
        if self._periodic_axes:
            raise ValueError("solver='adi' does not support manual periodic axes yet")
        if self._dft_planes:
            raise ValueError("solver='adi' does not support DFT plane probes yet")
        if self._ntff is not None:
            raise ValueError("solver='adi' does not support NTFF accumulation yet")
        if self._coaxial_ports:
            raise ValueError("solver='adi' does not support coaxial ports yet")
        if self._lumped_rlc:
            raise ValueError("solver='adi' does not support lumped RLC elements yet")
        if self._thin_conductors:
            raise ValueError("solver='adi' does not support thin-conductor corrections yet")
        if debye_spec is not None or lorentz_spec is not None:
            raise ValueError("solver='adi' does not support dispersive materials yet")
        # Conductivity is now supported: implicit sigma in ADI tridiagonal.
        # Internal absorbing layers also use sigma, so no restriction needed.
        for pe in self._ports:
            if pe.impedance != 0.0 or pe.extent is not None:
                raise ValueError("solver='adi' currently supports only add_source()-style soft sources")
            if self._mode == "2d_tmz" and pe.component != "ez":
                raise ValueError("solver='adi' in 2D TMz mode supports only Ez soft sources")
        _valid_adi_probes = {"ez", "hx", "hy"} if self._mode == "2d_tmz" else {"ex", "ey", "ez", "hx", "hy", "hz"}
        for probe in self._probes:
            if probe.component not in _valid_adi_probes:
                raise ValueError(f"solver='adi' supports probes on {_valid_adi_probes} only")
