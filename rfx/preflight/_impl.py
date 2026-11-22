"""Run the preflight checks; execution may lend its conductors object for this call only.

The solve object is lent to an isolated reader for this call only. Standalone
preflight keeps using the diagnostic audit cache.
"""
from __future__ import annotations

from rfx._warning_origin import user_stacklevel
from rfx.errors import AccuracyWarning, RfxRefusal

import jax

from rfx.api._ad_memory_types import ADMemoryUnavailable
from rfx.preflight._common import (
    PreflightErrorWarning, PreflightIssue, PreflightReport, PreflightWarning,
)
from rfx.model.conductors import prepare_solve
from rfx.preflight._not_judged import (
    judge_or_report, preflight_scope, reporting_to,
)


def _preflight_impl_body(
    self,
    *,
    strict: bool = False,
    check_ntff: bool | str = True,
    check_resolution: bool = True,
    check_ad_memory: bool = False,
    n_steps_for_memory: int | None = None,
    available_memory_gb: float | None = None,
    _conductors=None,
    _tfsf_material_overrides=(),
    _conformal_run=None,
    _traced_entry=None,
    _conductor_operator=None,
) -> "PreflightReport":
    """Shared checks; execution lends its solve products only to this call."""
    import warnings
    from rfx.boundaries.model import refresh_boundary_model
    refresh_boundary_model(self)
    if getattr(self, "_waveguide_ports", ()) and _tfsf_material_overrides:
        import copy
        self = copy.copy(self)
        self._waveguide_call_overrides = _tfsf_material_overrides
    if _conformal_run is not None or _conductor_operator is not None:
        import copy
        from rfx.boundaries.features import share_wall_warning_history
        run_copy = copy.copy(self)
        share_wall_warning_history(self, run_copy)
        self = run_copy
        if _conformal_run is not None:
            self._conformal_run_call = bool(_conformal_run)
        self._conductor_operator_call = _conductor_operator
    if _conductors is not None:
        import copy
        import weakref
        from rfx.preflight.realization import _CampaignStaticsContext
        from rfx.boundaries.features import share_wall_warning_history
        diagnostic_copy = copy.copy(self)
        share_wall_warning_history(self, diagnostic_copy)
        self = diagnostic_copy
        with jax.ensure_compile_time_eval():
            ctx = _CampaignStaticsContext(self)
        ctx.sim = weakref.proxy(self)
        ctx._realized = _conductors
        ctx._assembly_lent = True
        self._campaign_ctx = lambda: ctx
        self._assemble_realized = lambda *a, **k: _conductors
    self._resolve_mesh()
    from rfx.preflight._subjects import finding_record
    issues = PreflightReport()
    verdict_findings = []
    if _conductors is not None:
        ref = weakref.ref(self)
        self._finding_record_reader = lambda: _conductors._realization_record or finding_record(ref())
        tables = _conductors._realization_tables
        issues._realization_reader = lambda: tables[-1] if tables else None
        if tables:
            verdict_findings.extend(tables[-1].findings)
    else:
        issues._realization_reader = lambda: self.realized_geometry().realization_table

    from rfx.preflight.realization import assembly_warning_scope
    with warnings.catch_warnings(record=True) as caught, jax.ensure_compile_time_eval(), assembly_warning_scope():
        warnings.simplefilter("always")
        from rfx.preflight._msl_probe_rules import admit_msl_auto_probes
        admit_msl_auto_probes(self)
        from rfx.measurement import admit_waveguide_entries
        admit_waveguide_entries(self)
        self._collect_flux_regions(issues)
        try:
            if _conductors is None:
                from rfx.model.realization_verdict import admit_table
                ctx = self._campaign_ctx()
                root = ctx.bind_realization()
                if root is not None:
                    admit_table(self, root, deliver=False)
                    admit_table(self, root, finished=True, deliver=False)
                    table = root.realization_table
                    setattr(issues, "_realization_reader", lambda: table)
                    verdict_findings.extend(table.findings)
        except ValueError as overlap_error:
            if getattr(overlap_error, "code", None) != "absorber_overlap":
                raise
            from typing import cast
            verdict_findings.append(PreflightIssue(str(overlap_error), severity="error",
                code=cast(str, getattr(overlap_error, "code")),
                source=cast(str | None, getattr(overlap_error, "source"))))
        try:
            from rfx.model.thin_conductors import warn_dc_films
            judge_or_report("warn_dc_films", lambda: warn_dc_films(self, warnings), issues)
            if check_resolution:
                judge_or_report("_validate_mesh_quality", self._validate_mesh_quality, issues)
            judge_or_report("_validate_simulation_config", self._validate_simulation_config, issues)
            from rfx.boundaries.tfsf import report
            judge_or_report(
                "report", lambda: report(self, issues, root=_conductors,
                    material_overrides=_tfsf_material_overrides), issues)
            from rfx.model.pad_fill import report_pad_fill
            judge_or_report("report_pad_fill", lambda: report_pad_fill(self, issues), issues)
            if check_ntff:
                judge_or_report(
                    "_validate_ntff_inverse_design", lambda: self._validate_ntff_inverse_design(
                        include_pec_overlap_error=(check_ntff != "advisory")),
                    issues)
        except ValueError as e:
            duplicate_dc = (getattr(e, 'source', None) == 'admit_dc_film'
                and any(getattr(w.message, 'source', None) == 'admit_dc_film'
                        and getattr(w.message, 'loc', None) is not None
                        and str(w.message) == str(e) for w in caught))
            from rfx.preflight.realization import _same_assembly_failure
            duplicate_assembly = (
                not getattr(e, 'code', None)
                and any(getattr(w.message, 'code', None) in
                        ('assembly_refusal', 'assembly_fault') for w in caught)
                and _same_assembly_failure(self, e)
            )
            if not duplicate_dc and not duplicate_assembly:
                issues.append(PreflightIssue(
                    f"ERROR: {e}",
                    severity="error",
                    code=getattr(e, "code", "uncoded"),
                    loc=getattr(e, "loc", None),
                    source=getattr(e, "source", None),
                    diagnostic=getattr(e, "diagnostic", None),
                    scope=getattr(e, "scope", "model"),
                    subjects=getattr(e, "_subjects", ()),
                    region=getattr(e, "_region", None),
                ))

    dc_findings = {str(w.message) for w in caught
                   if getattr(w.message, 'source', None) == 'admit_dc_film'
                   and getattr(w.message, 'loc', None) is not None}
    verdict_issues = tuple(issues)
    for w in caught:
        msg = str(w.message)
        if any(issue.code == getattr(w.message, 'code', None) and str(issue) == msg
               for issue in verdict_issues):
            continue
        if (getattr(w.message, 'source', None) == 'admit_dc_film'
                and getattr(w.message, 'loc', None) is None and msg in dc_findings):
            continue
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
        issues.append(
            PreflightIssue(
                msg,
                severity=severity,
                code=code,
                loc=loc,
                source=source,
                diagnostic=getattr(inst, "diagnostic", None),
                scope=getattr(inst, "scope", "model"),
                subjects=getattr(inst, "_subjects", ()),
                region=getattr(inst, "_region", None),
            )
        )

    for finding in verdict_findings:
        if finding.code in {"sheet_effective_size", "sheet_cavity_spacing", "port_aperture_snap"}:
            for index, issue in enumerate(issues):
                if issue.code == finding.code and str(issue) == str(finding):
                    issues[index] = finding
            if any(issue.code == finding.code for issue in issues):
                continue
        if not any(issue.code == finding.code and str(issue) == str(finding) for issue in issues):
            issues.append(finding)

    if check_ad_memory:
        if n_steps_for_memory is None:
            raise ValueError("check_ad_memory=True requires n_steps_for_memory")
        try:
            est = self.estimate_ad_memory(
                n_steps_for_memory,
                available_memory_gb=available_memory_gb,
            )
        except ADMemoryUnavailable as error:
            issues.append(PreflightIssue(str(error), severity="error", code="ad_memory"))
        else:
            if est.warning:
                issues.append(PreflightIssue(
                    est.warning, severity="warning", code="ad_memory"
                ))

    assembly_codes = {"assembly_refusal", "assembly_fault"}
    located_refusal_codes = {
        "pec_box_subcell", "pec_zero_cells", "pec_realization_refused",
        "dc_film_area", "dc_film_layers", "dc_film_unchecked",
        "source_decoupled", "declared-span-short-of-padded-face",
        "port_aperture_unrasterizable", "absorber_point_refused",
    }
    if any(issue.code in assembly_codes for issue in issues):
        exc = self._campaign_ctx()._assembly_exception
        if exc is not None and any(
            issue.code in located_refusal_codes and issue.severity == "error"
            and str(exc) in str(issue) for issue in issues
        ):
            issues[:] = [issue for issue in issues if issue.code not in assembly_codes]

    from rfx.preflight._subjects import resolve_findings
    with reporting_to(issues):
        resolve_findings(issues)
    if strict and len(issues):
        raise issues.exception(
            f"preflight (strict) found {len(issues)} issue(s):\n  - "
            + "\n  - ".join(issues)
        )

    if len(issues):
        for iss in issues:
            print(f"  [PREFLIGHT] {iss}")
    elif check_ntff is True:
        print("  [PREFLIGHT] All checks passed.")
    elif check_ntff == "advisory":
        print("  [PREFLIGHT] All checks passed (NTFF advisory tier; the "
              "PEC-overlap error check runs on forward()/preflight()).")
    else:
        print("  [PREFLIGHT] All checks passed (NTFF checks skipped; "
              "run sim.preflight() for the full set).")

    if issues.flux_regions:
        from rfx.probes.flux_region import flux_region_message
        for record in issues.flux_regions:
            print(f"  [FLUX REGION] {flux_region_message(record)}")

    return issues


def _preflight_impl(
    self,
    *,
    strict: bool = False,
    check_ntff: bool | str = True,
    check_resolution: bool = True,
    check_ad_memory: bool = False,
    n_steps_for_memory: int | None = None,
    available_memory_gb: float | None = None,
    _conductors=None,
    _tfsf_material_overrides=(),
    _conformal_run=None,
    _traced_entry=None,
    _conductor_operator=None,
) -> "PreflightReport":
    """Scope traced-input names and reporting to this preflight call."""
    with preflight_scope(self, {} if _traced_entry is None else _traced_entry):
        return _preflight_impl_body(
            self, strict=strict, check_ntff=check_ntff,
            check_resolution=check_resolution, check_ad_memory=check_ad_memory,
            n_steps_for_memory=n_steps_for_memory, available_memory_gb=available_memory_gb,
            _conductors=_conductors, _tfsf_material_overrides=_tfsf_material_overrides,
            _conformal_run=_conformal_run, _traced_entry=_traced_entry,
            _conductor_operator=_conductor_operator,
        )


_preflight_impl.__qualname__ = "_PreflightMixin._preflight_impl"


def run_preflight_gate(issues, context: str, stacklevel: int = 3) -> None:
    """Apply the same warning/error policy to full or scoped preflight.

    The advisory names the first caller frame outside rfx, however many rfx frames lie between;
    ``stacklevel`` is accepted from older callers and not used.
    """
    issues = PreflightReport(i for i in issues if not getattr(i, "_realization_delivered", False))
    if not len(issues):
        return
    import warnings
    errors = [i for i in issues if getattr(i, "severity", "warning") == "error"]
    warns = [i for i in issues if getattr(i, "severity", "warning") != "error"]
    if warns:
        body = "\n  - ".join(warns)
        warnings.warn(
            f"[{context}] preflight found {len(warns)} advisory issue(s) - "
            f"pass skip_preflight=True to suppress:\n  - {body}",
            AccuracyWarning,
            stacklevel=user_stacklevel(),
        )
    if errors:
        detail = "\n  - ".join(errors)
        from rfx._diagnostic_transport import diagnostic_refusal
        realized_refusals = sum(getattr(i, "source", None) == "judge_realization"
                          or getattr(i, "code", None) == "sheet_effective_size" for i in errors)
        hint = ("pass skip_preflight=True to bypass" if not realized_refusals else
                ("skip_preflight=True bypasses only the non-realization errors; "
                 if realized_refusals < len(errors) else "") +
                "skip_preflight=True does not bypass a realization refusal: "
                'use Simulation(..., snap="declared") to report it as a warning, '
                "or apply the remedy in the message")

        raise diagnostic_refusal(
            RfxRefusal(
                f"[{context}] preflight found {len(errors)} blocking error(s) "
                f"({hint}):\n  - {detail}"
            ),
            issues.diagnostics,
            causes=PreflightReport(errors).diagnostics,
        )


def auto_preflight(sim, *, skip=False, context="forward", check_ntff=True,
                   conductors=None, prepare=False, distributed=False, tfsf_material_overrides=(),
                   conformal=None, realization_options=None):
    """Emit a UserWarning if preflight finds issues.

    Called automatically at the start of ``forward()``, ``optimize()``,
    and ``topology_optimize()`` so users discover physics violations
    (under-resolved mesh, geometry in CPML, probe in PEC, ...) before
    spending minutes of GPU compute. Pass ``skip_preflight=True`` at
    the call site to opt out (tests, already-validated configs).

    ``check_ntff`` gates the NTFF check family. ``run()`` passes
    ``check_ntff="advisory"``: the λ/4 near-field gap
    and sub-wavelength ground-plane advisories are
    physics-relevant to any far-field computation and
    run() is the common NTFF entry point, so they are surfaced as
    warnings — but only ``forward(port_s11_freqs=...)`` / ``optimize``
    (the inverse-design entry points) hard-fail on the PEC-overlap
    error. Previously run() skipped the family entirely and still
    printed "All checks passed", which contradicted explicit
    ``sim.preflight()`` output on the same configuration.

    ``prepare=True`` also returns an execution-local solve handoff. With
    skipped preflight it stays lazy until lane admission; otherwise checks
    read its assembled product before the selected runner takes ownership.
    """
    assembly = prepare_solve(sim, distributed=distributed, realization_options=realization_options) if prepare else None
    if skip:
        return assembly if prepare else PreflightReport()
    if assembly is not None:
        conductors = assembly.realized()
    traced_entry = dict(zip(
        ("eps_override", "sigma_override", "mu_r_override"), tfsf_material_overrides))
    from rfx.model.source_admission import conductor_operator
    smoothing, resolved_conformal = realization_options or (False, conformal)
    if resolved_conformal is None:
        resolved_conformal = bool(sim._boundary_spec.conformal_faces())
    operator = conductor_operator(smoothing=smoothing, conformal=resolved_conformal)
    if (getattr(sim, "_tfsf", None) is not None or sim._waveguide_ports) and any(v is not None for v in tfsf_material_overrides):
        issues = sim._preflight_impl(strict=False, check_ntff=check_ntff,
            _conductors=conductors, _tfsf_material_overrides=tfsf_material_overrides,
            _conformal_run=conformal, _traced_entry=traced_entry,
            _conductor_operator=operator)
    else:
        issues = sim._preflight_impl(strict=False, check_ntff=check_ntff,
                                     _conductors=conductors, _conformal_run=conformal,
                                     _traced_entry=traced_entry, _conductor_operator=operator)
    sim._run_preflight_gate(issues, context=context, stacklevel=5)
    if assembly is not None:
        assembly.diagnostics = tuple(getattr(issues, "diagnostics", ()))
    return assembly if prepare else issues
