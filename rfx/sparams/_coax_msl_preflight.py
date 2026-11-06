"""Shared sheet verdict delivery for the coax-MSL calculator."""


def coax_msl_sheet_preflight(self, skip_preflight):
    sheet_findings = None
    if not skip_preflight:
        # This lane owns its port checks; run only the shared sheet verdict.
        import warnings
        from rfx.geometry.csg import Box
        from rfx.preflight._common import PreflightIssue, PreflightWarning
        from rfx.preflight.pec_geometry import (
            _warn_campaign_statics_unavailable, _warn_sheet_effective_size,
        )

        with warnings.catch_warnings(record=True) as findings:
            warnings.simplefilter("always")
            ctx = self._campaign_ctx()
            if ctx.error is not None:
                _warn_campaign_statics_unavailable(warnings, ctx.error)
            else:
                boxes = [e for e in ctx.interior_pec_entries()
                         if e.kind in ("volume", "sheet") and isinstance(e.shape, Box)]
                _warn_sheet_effective_size(warnings, ctx, boxes)
        for finding in findings:
            if not isinstance(finding.message, PreflightWarning):
                warnings.warn_explicit(
                    finding.message, finding.category, finding.filename,
                    finding.lineno, source=finding.source,
                )
        issues = tuple(PreflightIssue(f.message, severity=f.message.severity,
                                     code=f.message.code, source=f.message.source)
                       for f in findings if isinstance(f.message, PreflightWarning))
        self._run_preflight_gate(issues, context="compute_coax_msl_transition")
        if ctx.error is None:
            # Retain this gate's delivery state for the same board at admission.
            sheet_findings = tuple(f for f in issues if f.code == "sheet_effective_size")
            for finding in sheet_findings:
                finding._realization_delivered = True

    return sheet_findings
