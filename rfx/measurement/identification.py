"""Resolve ring-down identification channels before the time loop."""
import warnings

from rfx._grid_metric import field_index


def resolve_identification_probes(probes, grid, pads, add):
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
    if duplicates:
        warnings.warn(
            f"Duplicate ring-down identification probe indices {duplicates} "
            "(zero-based) resolve to the same cell and component and produce "
            "identical columns. Remove the repeated probes or move them to "
            "different cells.", UserWarning, stacklevel=3)
    return tuple(keys)
