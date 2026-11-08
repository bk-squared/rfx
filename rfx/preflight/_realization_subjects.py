"""Annotations captured from the table that owns a realization verdict."""
from types import SimpleNamespace

from rfx.preflight._subjects import _Region, _Selection


def retained_subjects(finding):
    """Preserve deferred annotations without reading a different realization."""
    return dict(scope=finding.scope, subjects=finding._subjects, region=finding._region)


def _selection(table, keys):
    owner = table.prepared() if getattr(table, "prepared", None) is not None else None
    record = getattr(owner, "_realization_record", None)
    view = SimpleNamespace(realization_table=table, nodes=getattr(record, "nodes", ()))
    sim = SimpleNamespace(_dx_profile=None, _dy_profile=None, _dz_profile=None)
    return _Selection(sim, keys, lambda: view)


def table_subjects(table, rows, *, axes=None):
    """Name the owning rows; subregion comparisons retain their parent identity."""
    selection = _selection(table, tuple(row.label for row in rows))
    bounds = None
    if axes is not None and len(axes) == 3:
        bounds = tuple(tuple(a.bounds_m[end] for a in axes) for end in (0, 1))
    region = None if any(row.traced_inputs for row in rows) else _Region(selection, bounds).resolve()
    return dict(scope="object", subjects=selection.resolve(), region=region)


def bind_retained_subjects(finding, table):
    """Freeze a retained site's selection against its prepared table."""
    if isinstance(finding._subjects, _Selection):
        selection = _selection(table, finding._subjects.keys)
        finding._subjects = selection.resolve()
        if isinstance(finding._region, _Region):
            finding._region = _Region(selection, finding._region.bounds).resolve()
