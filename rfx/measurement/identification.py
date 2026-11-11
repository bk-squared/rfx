"""Resolve ring-down identification channels before the time loop."""
from rfx.errors import ConfigurationWarning
from contextlib import contextmanager
from contextvars import ContextVar
import inspect
import warnings

from rfx._internal._grid_metric import field_index


def resolve_identification_probes(probes, grid, pads, add, *, warn_duplicates=True):
    """Keep declared columns and warn once for repeated cell/component keys."""
    keys = []
    groups = {}
    for number, (position, component) in enumerate(probes):
        for ax, coord in enumerate(position):
            lo = pads[ax]
            hi = grid.shape[ax] - int(getattr(grid, f"pad_{'xyz'[ax]}_hi", 0))
            if not (grid.node_of(ax, lo) <= coord <= grid.node_of(ax, hi - 1)):
                raise ValueError("identification probe position is outside the probeable interior (absorber pad or domain)")
        idx = tuple(int(v) for v in field_index(grid, position, component))
        key = (component, idx)
        keys.append(key)
        groups.setdefault(key, []).append(number)
        add(component, idx, wire=False)
    duplicates = [indices for indices in groups.values() if len(indices) > 1]
    if duplicates and warn_duplicates:
        frame = inspect.currentframe().f_back
        stacklevel = 2
        while frame is not None and frame.f_globals.get("__name__", "").startswith("rfx."):
            stacklevel += 1
            frame = frame.f_back
        del frame
        warnings.warn(
            f"Duplicate ring-down identification probe indices {duplicates} "
            "(zero-based) resolve to the same cell and component and produce "
            "identical columns. Remove the repeated probes or move them to "
            "different cells.", ConfigurationWarning, stacklevel=stacklevel)
    return tuple(keys)


# Temporary route until graded entries leave api/_execute.py; structure plan §11.
_ringdown_probes: ContextVar[tuple | None] = ContextVar("ringdown_probes", default=None)


@contextmanager
def ringdown_probes(probes, grid, keys, entry_point):
    """Lend probes to lanes sharing the declaration list, including drive copies."""
    from rfx.api._spec import _ProbeEntry
    from rfx.ringdown import _node_position
    entries = tuple(_ProbeEntry(position=_node_position(grid, idx), component=comp,
                                allow_in_absorber=True) for comp, idx in keys)
    token = _ringdown_probes.set((probes, entries, entry_point))
    try:
        yield
    finally:
        _ringdown_probes.reset(token)


def read_ringdown_probes(probes, reader):
    """Return scoped entries; a changed declaration-list identity is an error."""
    context = _ringdown_probes.get()
    if context is None:
        return ()
    key, entries, entry_point = context
    if key is not probes:
        raise RuntimeError(f"{entry_point}: {reader} missed the ring-down probe-list key")
    return entries
