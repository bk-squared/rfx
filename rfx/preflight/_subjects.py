"""Object identities and grid regions carried by preflight findings.

Selection is captured at the check site and resolved after warning capture.
Unresolvable subjects remain contract defects, never legality findings.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from collections.abc import Callable, Sequence
from typing import Any, Protocol

import numpy as np


class _ProfileOwner(Protocol):
    _dx_profile: Any
    _dy_profile: Any
    _dz_profile: Any


_COLLECTIONS = {
    "_geometry": "geometry", "_thin_conductors": "thin_conductor",
    "_pinned_sheets": "pinned_sheet", "_ports": "port",
    "_waveguide_ports": "waveguide_port", "_msl_ports": "msl_port",
    "_coaxial_ports": "coaxial_port", "_floquet_ports": "floquet_port",
    "_probes": "probe", "_lumped_rlc": "lumped_rlc",
}


def label(record, key):
    """Read an existing label, including placements absent from entities."""
    table = getattr(record, "realization_table", None)
    rows = table.rows if table is not None else getattr(record, "rows", ())
    for row in (*getattr(record, "entities", ()), *rows,
                *getattr(record, "ports", ()), *getattr(record, "sources", ()),
                *getattr(record, "probes", ())):
        if row.label == key:
            return row.label
    raise ValueError(f"Finding subject {key!r} is absent from its realization record")


def _key(sim, obj):
    if isinstance(obj, str):
        return obj
    if hasattr(obj, "subject_label"):
        return obj.subject_label
    if hasattr(obj, "label"):
        return obj.label
    if isinstance(obj, tuple) and len(obj) == 2 and isinstance(obj[0], str) and obj[0] in _COLLECTIONS:
        return f"{_COLLECTIONS[obj[0]]}[{obj[1]}]"
    for collection, prefix in _COLLECTIONS.items():
        for i, entry in enumerate(getattr(sim, collection, ())):
            if obj is entry or obj is getattr(entry, "shape", None):
                return f"{prefix}[{i}]"
    for collection in ("_dft_planes", "_flux_monitors"):
        for entry in getattr(sim, collection, ()):
            if obj is entry:
                return entry.name
    for name in ("ntff", "tfsf", "current_moments"):
        if obj is getattr(sim, "_" + name, None):
            return name
    raise ValueError("Finding object is not a declaration in this model")


def finding_record(sim):
    """Read labels for findings even when public geometry cannot be produced.

    The identity-only fallback stays local to this reader: never publish it in
    Simulation's public geometry cache or return it from a public API.
    A refusal of the public record (ValueError, NotImplementedError) admits this
    fallback; the context owns it so repeated subject and region reads share one
    stand-in.
    """
    try:
        return sim.realized_geometry()
    except (ValueError, NotImplementedError) as exc:
        ctx = sim._campaign_ctx()
        from rfx.model.realization_unavailable import identity_record
        if getattr(ctx, '_finding_identity_record', None) is None:
            ctx._finding_identity_record = identity_record(sim, ctx, str(exc))
        return ctx._finding_identity_record


@dataclass
class _Selection:
    sim: _ProfileOwner
    keys: tuple[str, ...]
    reader: Callable[[], Any] | None = None

    def record(self):
        return self.reader() if self.reader is not None else finding_record(self.sim)

    def resolve(self):
        if not self.keys:
            return ()
        record = self.record()
        # Record order is declaration order, including holes in probe indices.
        table = getattr(record, "realization_table", None)
        rows = table.rows if table is not None else record.entities
        wanted = set(self.keys)
        result = tuple(row.label for row in rows if row.label in wanted)
        # Keep missing identities visible to the contract without changing the
        # original finding's code, severity, text or ordering.
        if wanted.difference(result):
            result += ("<unresolved finding subject>",)
        return tuple(dict.fromkeys(result))


@dataclass
class _Region:
    selection: _Selection
    bounds: Sequence[Sequence[float]] | None

    def resolve(self):
        from rfx.core.jax_utils import is_tracer
        sim = self.selection.sim
        if any(is_tracer(p) for p in (sim._dx_profile, sim._dy_profile, sim._dz_profile)):
            return None
        record = self.selection.record()
        if not record.nodes:
            return None
        bounds = self.bounds
        if bounds is None:
            table = record.realization_table
            axes = [table.row(key).axes for key in self.selection.keys]
            if not axes or any(len(row) != 3 for row in axes):
                return None
            bounds = (tuple(min(row[a].bounds_m[0] for row in axes) for a in range(3)),
                      tuple(max(row[a].bounds_m[1] for row in axes) for a in range(3)))
        lo, hi = bounds
        cells = []
        for a, nodes in enumerate(record.nodes):
            nodes = np.asarray(nodes)
            start = max(0, min(len(nodes)-1, int(np.searchsorted(nodes, lo[a], side="right"))-1))
            stop = max(start, min(len(nodes)-1, int(np.searchsorted(nodes, hi[a], side="left"))))
            cells.append((start, stop))
        return {"bounds_m": (tuple(float(record.nodes[a][cells[a][0]]) for a in range(3)),
                              tuple(float(record.nodes[a][cells[a][1]]) for a in range(3))),
                "cells": tuple(cells)}


def finding_subjects(ctx, *, objects=(), region=None):
    """Capture declarations or record rows, never message text or locations."""
    sim = getattr(ctx, "sim", ctx)
    keys = []
    for obj in objects:
        try:
            keys.append(_key(sim, obj))
        except Exception:
            keys.append("<unresolved finding subject>")
    selection = _Selection(sim, tuple(dict.fromkeys(keys)),
                           getattr(sim, "_finding_record_reader", None))
    return {"scope": "object" if selection.keys else "place" if region is not None else "model",
            "subjects": selection,
            "region": _Region(selection, region) if region is not None or selection.keys else None}


def resolve_findings(issues):
    """Freeze plain values from the judged record, outside warning capture.

    Annotation must never replace a finding or fail preflight. Missing labels
    and resolution faults are exposed as invalid subjects to the contract.
    """
    records = {}
    for issue in issues:
        selection = issue._subjects
        if not isinstance(selection, _Selection):
            continue
        try:
            if selection.keys or issue._region is not None:
                key = id(selection.sim)
                if key not in records:
                    records[key] = _finding_view(selection.sim, selection.record())
                selection.reader = lambda record=records[key]: record
            issue._subjects = selection.resolve()
        except Exception:
            issue._subjects = ("<unresolved finding subject>",)
            issue._region = None
        try:
            issue._region = issue._region.resolve() if issue._region is not None else None
        except Exception:
            issue._region = None


def _finding_view(sim, record):
    """Add identity-only placements privately for traced and refused records."""
    table = record.realization_table
    domain = table.row("domain")
    if domain.axes:
        return record
    from rfx.model.realization_placements import traced_placement_rows, label_rows
    # Keep the identity reader's existing declaration order. The public traced
    # table retains main's geometry/domain/probe rows, without these additions.
    rows = tuple(row for row in table.rows if not row.label.startswith("probe["))
    rows += tuple(traced_placement_rows(sim, domain.status, domain.traced_inputs))
    rows += tuple(label_rows(sim))
    return replace(record, realization_table=replace(table, rows=rows))


def public_probes(sim):
    """Exclude internal witnesses from user-facing placement findings."""
    internal = getattr(sim, "_internal_probe_indices", None) or ()
    return tuple(p for i, p in enumerate(sim._probes) if i not in internal)


def absorber_region(sim, faces):
    """Bounding box of the affected exterior slabs on the padded grid."""
    ctx = sim._campaign_ctx()
    boxes = []
    for face in faces:
        a, side = "xyz".index(face[0]), face[-2:]
        lo = [float(n[0]) for n in ctx.nodes]
        hi = [float(n[-1]) for n in ctx.nodes]
        depth = int(getattr(ctx.grid, f"pad_{face}"))
        if side == "lo":
            hi[a] = float(ctx.nodes[a][depth])
        else:
            lo[a] = float(ctx.nodes[a][len(ctx.nodes[a])-1-depth])
        boxes.append((lo, hi))
    return (tuple(min(box[0][a] for box in boxes) for a in range(3)),
            tuple(max(box[1][a] for box in boxes) for a in range(3)))


def runway_region(sim, axis, side, runway):
    """Interior cells used by the grading check, in physical coordinates."""
    a = "xyz".index(axis)
    lo, hi = [0.] * 3, list(sim._domain)
    width = float(np.sum(runway))
    if side == "lo":
        hi[a] = width
    else:
        lo[a] = hi[a] - width
    return tuple(lo), tuple(hi)


class SubjectMessage(str):
    """Keep declaration owners alongside a backwards-compatible message."""

    objects: tuple[object, ...]

    def __new__(cls, message, objects):
        result = super().__new__(cls, message)
        result.objects = tuple(objects)
        return result
