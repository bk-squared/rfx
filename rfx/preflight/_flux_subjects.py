"""Finite flux-window findings and their declaration identities."""
from __future__ import annotations

import jax

from rfx.core.jax_utils import is_tracer
from rfx.preflight._common import PreflightIssue, PreflightReport


def _collect_flux_regions(self, report: PreflightReport) -> None:
    """Record the same finite windows the uniform/NU runners consume.

    Only geometry is evaluated, never material arrays or field values.
    Traced geometry cannot provide metre bounds; keep that limitation
    explicit without forcing a material-AD caller off its tape.
    """
    from rfx.probes.flux_region import (
        flux_region_message, resolve_flux_region, validate_flux_region_inputs,
    )

    source = "_collect_flux_regions"

    def finding(entry, message, code, severity="warning"):
        from rfx.preflight._subjects import finding_subjects
        report.append(PreflightIssue(
            message, severity=severity, code=code,
            **finding_subjects(self, objects=(entry,)),
            loc=getattr(entry, "name", None), source=source,
        ))

    def traced(value):
        return any(is_tracer(leaf) for leaf in jax.tree_util.tree_leaves(value))

    finite = []
    for entry in getattr(self, "_flux_monitors", ()):
        geometry = (entry.size, entry.center, entry.coordinate)
        if traced(geometry):
            finding(entry, "Flux monitor region is unavailable: traced geometry.",
                    "flux_region_unavailable")
            continue
        try:
            size, _ = validate_flux_region_inputs(entry.size, entry.center)
        except (ValueError, TypeError) as exc:
            finding(entry, f"Flux monitor {entry.name!r}: {exc}",
                    "flux_region_invalid", "error")
            continue
        if size is not None:
            finite.append(entry)
    if not finite:
        return

    # Resolve the selected/frozen mesh, including inferred NU profiles.
    # No independent rounding or declaration-only grid choice belongs here.
    try:
        if traced(self._resolve_mesh()):
            raise ValueError("traced mesh has no concrete metre bounds")
        grid = self._build_realized_grid()
    except (ValueError, TypeError, NotImplementedError, AttributeError,
            IndexError, KeyError) as exc:
        for entry in finite:
            finding(entry, f"Flux monitor {entry.name!r} region is unavailable: {exc}",
                    "flux_region_unavailable")
        return

    for entry in finite:
        try:
            record = resolve_flux_region(grid, entry, self._domain, warn=False)
        except (ValueError, TypeError, IndexError) as exc:
            finding(entry, f"Flux monitor {entry.name!r}: {exc}",
                    "flux_region_invalid", "error")
            continue
        report.flux_regions.append(record)
        if record["clamped"]:
            finding(entry, flux_region_message(record), "flux_region_clamped")


_collect_flux_regions.__qualname__ = "_PreflightMixin._collect_flux_regions"
