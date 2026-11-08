"""Admission of the comparison rows produced by a prepared solve.

The table owns execution-local judgement and delivery state. Readers may inspect
the findings repeatedly; only newly available rows are judged at a hand-off.
"""
from __future__ import annotations

from types import SimpleNamespace

from rfx.preflight._realization_subjects import (
    bind_retained_subjects, retained_subjects, table_subjects,
)


def _mark_port_built(table):
    """Keep execution and diagnostic rows explicit about their port owner."""
    from dataclasses import replace
    # Soft point sources share port labels but do not build conductors.
    port_labels = {row.label for row in table.rows
                   if row.label.startswith(("port[", "coaxial_port["))
                   and row.family != "soft_point"}
    object.__setattr__(table, "rows", tuple(
        replace(row, status="not judged here: port-built (judged by the port)")
        if row.label in port_labels else row for row in table.rows))
    table.judged.update(port_labels)
    return port_labels


def judge_realization(table, snap):
    """Return findings for rows that have not yet been judged in this solve."""
    from rfx.preflight._common import PreflightIssue
    from dataclasses import replace
    findings = []
    port_labels = _mark_port_built(table)
    reason = getattr(table, "not_evaluated_reason", None)
    if reason is not None:
        object.__setattr__(table, "rows", tuple(
            replace(row, status=f"not evaluated: {reason}")
            if row.label not in port_labels else row for row in table.rows))
        table.judged.update(row.label for row in table.rows)
        if reason not in table.judged:
            table.judged.add(reason)
            findings.append(PreflightIssue(
                f"realization: not evaluated: {reason}; geometric comparisons were skipped. "
                "Evaluate on uniform run() to check the realized dimensions.",
                code="realization", source="judge_realization",
                **table_subjects(table, tuple(r for r in table.rows if r.model != "label only"))))
        table.findings.extend(findings)
        return tuple(findings)
    rows = [(row, row) for row in (*table.rows, *getattr(table, "legacy_rows", ()))]
    for row in table.rows:
        for region, axes in getattr(row, "regions", ()):
            rows.append((replace(row, label=f"{row.label}/{region}", axes=axes, regions=()), row))
    for row, owner in rows:
        if row.label in table.judged or row.label in table.deferred:
            continue
        if row.status.startswith("not evaluated: traced") or row.traced_inputs:
            table.judged.add(row.label)
            if "traced" not in table.judged:
                table.judged.add("traced")
                findings.append(PreflightIssue(
                    "realization: not evaluated: traced; geometric comparisons were skipped. "
                    "Evaluate a concrete mesh and geometry to check the realized dimensions.",
                    code="realization", source="judge_realization",
                    **table_subjects(table, tuple(r for r in table.rows
                        if r.status.startswith("not evaluated: traced") or r.traced_inputs))))
            continue
        if row.status == "not judged: edge not measured":
            # DESIGN.md D:c1d171 / D:8ea1e2: node span is not a measured edge.
            table.judged.add(row.label)
            findings.append(PreflightIssue(
                f"realization {row.label} ({row.family}): not judged: edge not measured; "
                "the node span is reported without a solved extent comparison.",
                severity="warning", code="realization", loc=row.label, source="judge_realization",
                **table_subjects(table, (owner,), axes=row.axes)))
            continue
        if row.status == "not judged: no reference":
            table.judged.add(row.label)
            findings.append(PreflightIssue(
                f"realization {row.label} ({row.family}): not judged: no reference; "
                "the model integral is reported without a drawn integral comparison. "
                "Use geometry with a supported analytic reference to judge its realization.",
                code="realization", loc=row.label, source="judge_realization",
                **table_subjects(table, (owner,), axes=row.axes)))
            continue
        if row.status != "evaluated":
            continue
        table.judged.add(row.label)
        if row.family == "legacy":
            if row.relative_miss is None or abs(row.relative_miss) > row.tolerance:
                findings.append(PreflightIssue(row.message, code=row.code,
                    severity="warning" if snap == "declared" else "error",
                    source="judge_realization"))
            continue
        axes = tuple(a for a in row.axes if a.declared_span_m is not None
                     and a.axis in getattr(table, "active_axes", "xyz"))
        # Point placement is judged only by interior admission; retain its shift
        # in the table without treating node rounding as an extent-object warning.
        centroid = any(a.declared_span_m > 0 for a in axes) and any(
            a.centroid_shift_cells is not None and
            abs(a.centroid_shift_cells) > .25 for a in axes)
        if row.family == "domain":
            centroid = False
            axes = tuple(a for a in axes if a.axis not in table.early_domain_axes)
        if row.volume_miss_percent is not None or row.open_area_m2 is not None:
            centroid = False
        if row.integral_comparison is not None:
            measure, declared, realized, dimension = row.integral_comparison
            cell = sum(a.cell_size_m for a in axes)/len(axes) if axes else 1.
            percent = 100*((realized/declared)**(1/dimension)-1)
            miss = (realized**(1/dimension)-declared**(1/dimension))/cell
            quantity = f"{measure}-equivalent length miss {miss:+.12g} cells ({percent:+.12g} %)"
            mismatch = abs(percent) > 1.
        elif row.volume_miss_percent is not None:
            percent = row.volume_miss_percent
            cell = sum(a.cell_size_m for a in axes) / len(axes) if axes else 1.
            miss = (row.volume_equivalent_length_m - row.declared_volume_m3 ** (1/3)) / cell
            quantity = f"filled volume-equivalent length miss {miss:+.12g} cells ({percent:+.12g} %)"
            mismatch = abs(percent) > 1.
        elif row.area_miss_percent is not None:
            percent = row.area_miss_percent
            measure = "summed open area" if row.open_area_m2 is not None else "area"
            quantity = (f"{measure}-equivalent length miss {row.area_miss_cells:+.12g} cells "
                        f"({percent:+.12g} %)")
            mismatch = abs(percent) > 1.
        else:
            mismatch = any(a.miss_percent is not None and abs(a.miss_percent) > 1.
                           for a in axes)
            quantity = "; ".join(
                f"{a.axis} extent miss {a.miss_cells:+.12g} cells ({a.miss_percent:+.12g} %)"
                for a in axes if a.miss_percent is not None)
        if ((row.model == "solved sheet edges" and row.shape_name == "Box") or
                row.model == "cell-centre footprint; node plane"):
            mismatch = False  # Existing film admission already judged its size.
        if not mismatch and not centroid:
            continue
        detail = "; ".join(
            f"{a.axis}: realized {a.realized_span_m/a.cell_size_m:.12g} cells "
            f"({a.realized_span_m:.12g} m), declared "
            f"{a.declared_span_m/a.cell_size_m:.12g} cells ({a.declared_span_m:.12g} m), "
            f"centroid shift {a.centroid_shift_cells:+.12g} cells "
            f"({a.centroid_shift_cells*a.cell_size_m:+.12g} m)"
            for a in axes if a.centroid_shift_cells is not None)
        dielectric = row.family == "material"
        severity = "error" if mismatch and not dielectric and snap == "strict" else "warning"
        remedy = ("Refine the mesh or change the drawn dimensions."
                  if dielectric else
                  "Move the edge to a node, refine, or use rfx.mesh_edges.edge_aware_profiles "
                  "to place nodes on the drawn edges.")
        if not mismatch:
            quantity = "centroid shift exceeds 0.25 cell"
        for a in axes:
            if a.declared_intervals_m:
                detail += (f"; {a.axis} periodic union declared {a.declared_intervals_m} m, "
                           f"realized {a.realized_intervals_m} m")
        if row.clipped_faces:
            detail += "; clipped faces " + ", ".join(
                f"{face}: {drawn:.12g} -> {wall:.12g} m" for face, drawn, wall in row.clipped_faces)
        if row.max_owner_deviation is not None:
            detail += f"; maximum per-owner face deviation {row.max_owner_deviation:.12g} (reported only)"
        finding = PreflightIssue(
            f"realization {row.label} ({row.family}): {quantity}; {detail}. {remedy}",
            severity=severity, code=("port_aperture_snap" if row.label.startswith("waveguide_port[") else "realization"),
            loc=row.label, source="judge_realization",
            **table_subjects(table, (owner,), axes=row.axes))
        setattr(finding, "_realization_axes", tuple(a.axis for a in axes
            if a.miss_percent is not None and abs(a.miss_percent) > 1.))
        base_label = row.label.split("/", 1)[0]
        previous = next((f for f in table.findings + findings
                         if f.code == "realization" and f.loc == base_label), None)
        if previous is not None and row.family in ("material", "volume"):
            merged = PreflightIssue(f"{previous} {finding}", severity=severity,
                code="realization", loc=base_label, source="judge_realization",
                **table_subjects(table, (owner,)))
            target = table.findings if previous in table.findings else findings
            target[target.index(previous)] = merged
            if getattr(previous, "_realization_delivered", False) and severity == "warning":
                setattr(merged, "_realization_delivered", True)
            elif target is table.findings:
                table.findings.remove(merged)
                findings.append(merged)
            continue
        findings.append(finding)
    table.findings.extend(findings)
    return tuple(findings)


def legacy_mismatch(relative_miss, tolerance):
    """A comparison available before the prepared table, using the same judge.

    Legacy callers retain their exception types, finding codes and prose. Their
    already admitted dimensions are excluded when the solve table is bound.
    """
    row = SimpleNamespace(label="legacy", family="legacy", status="evaluated",
                          traced_inputs=(), relative_miss=relative_miss,
                          tolerance=tolerance, message="", code="realization")
    table = SimpleNamespace(rows=(row,), judged=set(), deferred=set(), findings=[])
    return bool(judge_realization(table, "strict"))


def legacy_table_ready(ctx):
    root = getattr(ctx, "_realized", None)
    table = getattr(root, "realization_table", None)
    return table is not None and "legacy sheets" in table.judged


def admit_table(sim, conductors, *, options=None, finished=False, legacy_findings=None, deliver=True):
    """Judge only available rows, without moving any solver preparation stage."""
    from rfx.grid import axis_kinds
    from rfx.boundaries.features import grid_face_depths
    table = conductors.realization_table
    if table is None:
        return
    if table.report_only:
        _mark_port_built(table)
        return
    if table.not_evaluated_reason is not None:
        findings = judge_realization(table, sim._snap)
        if deliver:
            deliver_findings(table, findings)
        return
    smoothing, conformal = options or (False, False)
    kinds = axis_kinds(grid_face_depths(conductors.grid))
    table.early_domain_axes.update(axis for axis, kind in kinds.items()
        if getattr(sim, f"_d{axis}_profile") is None)
    if sim._dz_profile is not None:
        table.early_domain_axes.add("z")
    if not finished:
        for owner in (sim, conductors.grid, sim._unresolved_domain):
            for row, finding in early_outcomes(owner).values():
                if row.label.startswith("legacy:film-entry:"):
                    key = int(row.label.rsplit(":", 1)[1])
                    finding.loc = next((f"thin_conductors[{i}]" for i, tc in
                                        enumerate(sim._thin_conductors) if id(tc) == key), None)
                    owners = tuple(table.row(f"thin_conductor[{i}]") for i, tc in
                                   enumerate(sim._thin_conductors) if id(tc) == key)
                    annotations = table_subjects(table, owners)
                    finding.scope = annotations["scope"]
                    finding._subjects, finding._region = annotations["subjects"], annotations["region"]
                if row.label not in table.judged:
                    table.legacy_rows.append(row)
                    table.judged.add(row.label)
                    table.findings.append(finding)
        for row in table.rows:
            if row.traced_inputs:
                continue
            if (row.label.startswith(("port[", "waveguide_port[")) or
                    (row.label.startswith("geometry[") and
                     ((smoothing and row.family == "material") or
                      (conformal and row.family == "volume")))):
                table.deferred.add(row.label)
    else:
        table.deferred.clear()
    # Every lane's table passes here at least once (finish_table), whether or
    # not skip_preflight ran: the sheet-size rows are admitted on the first pass.
    legacy = admit_sheet_sizes(sim, conductors, legacy_findings, conformal)
    findings = tuple(legacy) + judge_realization(table, sim._snap)
    if deliver:
        deliver_findings(table, findings)


def admit_sheet_sizes(sim, conductors, legacy_findings, conformal):
    """Admit the PEC sheet and Box-sheet extent rows once per table.

    ``legacy_findings`` are rows a caller already formed (and may already have
    delivered); ``None`` means none were formed, so they are formed here.
    """
    table = conductors.realization_table
    if "legacy sheets" in table.judged:
        return []
    if legacy_findings is not None:
        table.judged.add("legacy sheets")
        for finding in legacy_findings:
            bind_retained_subjects(finding, table)
            table.findings.append(finding)
            retain_sheet_outcome(table, finding)
        return list(legacy_findings)
    from rfx.model.realization_reference import table_wide
    if table_wide(table.rows):
        return []
    from rfx.model.realization_context import context_from_conductors
    from rfx.preflight.pec_geometry import _warn_sheet_effective_size, _warn_sheet_cavity_spacing
    from rfx.geometry.csg import Box
    from rfx.preflight._common import PreflightIssue
    import numpy as np
    ctx = context_from_conductors(sim, conductors)
    ctx.cell_sizes = ctx.spacings
    ctx.nodes = tuple(n[:conductors.grid.shape[a]] for a, n in enumerate(ctx.nodes))
    from rfx.geometry.rasterize_grid import GridCoords
    import copy
    x, y, z = ctx.nodes
    ctx.coords = GridCoords(x, y, z, shape=conductors.grid.shape)
    ctx.sim = copy.copy(sim)
    ctx.sim._conformal_run_call = conformal
    entries = [e for e in ctx.interior_pec_entries()
               if e.kind in ("volume", "sheet") and e.lo is not None]
    legacy = []

    def collect(warning):
        legacy.append(PreflightIssue(str(warning), severity=warning.severity,
            code=warning.code, loc=warning.loc, source=warning.source,
            **retained_subjects(warning)))
        bind_retained_subjects(legacy[-1], table)

    collector = SimpleNamespace(warn=collect)
    _warn_sheet_effective_size(collector, ctx, [e for e in entries if isinstance(e.shape, Box)])
    pec = conductors.pec_mask
    _warn_sheet_cavity_spacing(collector, ctx, entries,
                              np.zeros(conductors.grid.shape, dtype=bool) if pec is None else pec)
    table.judged.add("legacy sheets")
    table.findings.extend(legacy)
    for finding in legacy:
        retain_sheet_outcome(table, finding)
    return legacy


def deliver_findings(table, findings):
    """Deliver newly formed findings once, before any stepping call."""
    import warnings
    from rfx._warning_origin import user_stacklevel
    from rfx.errors import RfxRefusal
    from rfx.preflight._common import PreflightWarning

    errors = []
    for finding in findings:
        if getattr(finding, "_realization_delivered", False):
            continue
        table.delivered.add(id(finding))
        finding._realization_delivered = True
        if finding.severity == "error":
            errors.append(str(finding))
        else:
            warnings.warn(PreflightWarning(str(finding), code=finding.code,
                severity=finding.severity, loc=finding.loc, source=finding.source,
                **retained_subjects(finding)),
                stacklevel=user_stacklevel())
    if errors:
        raise RfxRefusal("\n".join(errors))


def legacy_outcome(label, relative_miss, tolerance, message, code, source, *, loc=None):
    """Retain an early judge result with the legacy caller's rendered message."""
    from rfx.preflight._common import PreflightIssue
    row = SimpleNamespace(label=label, family="legacy", status="evaluated",
        traced_inputs=(), relative_miss=relative_miss, tolerance=tolerance,
        message=message, code=code)
    finding = PreflightIssue(message, code=code, source=source, loc=loc)
    setattr(finding, "_realization_delivered", True)
    return row, finding


def retain_sheet_outcome(table, finding):
    row, _ = legacy_outcome(f"legacy:{finding.code}", None, .01,
                           str(finding), finding.code, finding.source)
    table.legacy_rows.append(row)
    table.judged.add(row.label)


def remember_legacy(owner, key, outcome):
    early_outcomes(owner, create=True)[key] = outcome


def early_outcomes(owner, *, create=False):
    if hasattr(owner, "dx_arr"):
        from rfx.model.realization_integrals import quantities
        store = quantities(owner, create=create)
        return store.setdefault("early_outcomes", {}) if create else store.get("early_outcomes", {})
    outcomes = getattr(owner, "_realization_early_outcomes", None)
    if outcomes is None and create:
        outcomes = {}
        object.__setattr__(owner, "_realization_early_outcomes", outcomes)
    return outcomes if outcomes is not None else {}


def replay_legacy_findings(warnings, ctx, code):
    """Render retained outcomes at the original diagnostic position."""
    from rfx.preflight._common import PreflightWarning
    for finding in ctx._realized.realization_table.findings:
        if finding.code == code:
            warnings.warn(PreflightWarning(str(finding), code=finding.code,
                severity=finding.severity, loc=finding.loc, source=finding.source,
                **retained_subjects(finding)))
