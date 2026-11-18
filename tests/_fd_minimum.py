"""Exact smallest-cell witnesses for finite differences with derived dt."""

import numpy as np


def assert_same_minimum_cells(working, points):
    """Require every labelled FD profile to retain the working minimum set.

    Widths are compared exactly in the dtype supplied by the caller. The
    next-smallest width is the next distinct value, or None for a uniform
    profile. Report all mismatches so both signs of a crossing are visible.
    """
    def describe(vector):
        vector = np.asarray(vector)
        widths = np.unique(vector)
        indices = np.flatnonzero(vector == widths[0]).tolist()
        next_width = float(widths[1]) if len(widths) > 1 else None
        return indices, float(widths[0]), next_width

    base_set, base_min, base_next = describe(working)
    mismatches = []
    for label, vector in points:
        point_set, point_min, point_next = describe(vector)
        if point_set != base_set:
            mismatches.append(
                f"{label}: smallest-cell set {point_set} != working {base_set}; "
                f"point smallest={point_min!r}, next-smallest={point_next!r}; "
                f"working smallest={base_min!r}, next-smallest={base_next!r}")
    assert not mismatches, "\n".join(mismatches)
