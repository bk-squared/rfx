"""Run the preflight checks; execution may lend its conductors object for this call only.

The solve object is lent to an isolated reader for this call only. Standalone
preflight keeps using the diagnostic audit cache.
"""
from __future__ import annotations

import jax

from rfx.preflight._common import (
    PreflightErrorWarning, PreflightIssue, PreflightReport, PreflightWarning,
)


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
) -> "PreflightReport":
    """Shared checks; execution lends its solve products only to this call."""
    import warnings
    if _conductors is not None:
        # Execution lends its assembled, pre-port products for this call only.
        # Bind the legacy validators to an isolated reader, never the public
        # diagnostic cache (nor any instance override of that cache).
        import copy
        import weakref
        from rfx.preflight.realization import _CampaignStaticsContext
        self = copy.copy(self)
        with jax.ensure_compile_time_eval():
            ctx = _CampaignStaticsContext(self)
        ctx.sim = weakref.proxy(self)
        ctx._realized = _conductors
        self._campaign_ctx = lambda: ctx
        self._assemble_realized = lambda *a, **k: _conductors
    # Selection information belongs outside the captured legality findings.
    # Resolve before any validator reads a profile or fallback spacing.
    self._resolve_mesh()
    issues = PreflightReport()

    # Static geometry diagnostics must stay host-side under an outer
    # jit, just like mesh selection. Otherwise even a concrete Box's
    # mask becomes a tracer before validators convert it to numpy.
    # Incoming mesh/design tracers remain tracers and retain the
    # validators' existing not-evaluable guards.
    from rfx.preflight.realization import assembly_warning_scope
    with warnings.catch_warnings(record=True) as caught, jax.ensure_compile_time_eval(), assembly_warning_scope():
        warnings.simplefilter("always")
        self._collect_flux_regions(issues)
        try:
            from rfx.model.thin_conductors import warn_dc_films
            warn_dc_films(self, warnings)
            if check_resolution:
                self._validate_mesh_quality()
            self._validate_simulation_config()
            from rfx.boundaries.tfsf import report
            report(self, issues, root=_conductors, material_overrides=_tfsf_material_overrides)
            from rfx.model.pad_fill import report_pad_fill
            report_pad_fill(self, issues)
            if check_ntff:
                self._validate_ntff_inverse_design(
                    include_pec_overlap_error=(check_ntff != "advisory"),
                )
        except ValueError as e:
            # Collect (do NOT fail-on-first): the aggregated raise at the
            # end escalates every finding at once under strict.
            # Structurally-impossible configs raise PreflightConfigError
            # with the slug set at the check site; any other ValueError is
            # error-severity but uncoded.
            duplicate_dc = (getattr(e, 'source', None) == 'admit_dc_film'
                and any(getattr(w.message, 'source', None) == 'admit_dc_film'
                        and getattr(w.message, 'loc', None) is not None
                        and str(w.message) == str(e) for w in caught))
            if not duplicate_dc:
                issues.append(PreflightIssue(
                    f"ERROR: {e}",
                    severity="error",
                    code=getattr(e, "code", "uncoded"),
                    loc=getattr(e, "loc", None),
                    source=getattr(e, "source", None),
                ))

    # The explicit per-film reader owns DC findings. Diagnostic assembly may
    # replay the same warning, including from its cache. Preserve separate
    # declarations even when they share a shape and identical message text.
    dc_findings = {str(w.message) for w in caught
                   if getattr(w.message, 'source', None) == 'admit_dc_film'
                   and getattr(w.message, 'loc', None) is not None}
    for w in caught:
        msg = str(w.message)
        if (getattr(w.message, 'source', None) == 'admit_dc_film'
                and getattr(w.message, 'loc', None) is None and msg in dc_findings):
            continue
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

    if strict and len(issues):   # PreflightReport refuses bool() (#980)
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

    if len(issues):              # PreflightReport refuses bool() (#980)
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


# Restored so rfx/api/__init__.py rewrites it to "Simulation._preflight_impl" like every other moved preflight body.
_preflight_impl.__qualname__ = "_PreflightMixin._preflight_impl"
