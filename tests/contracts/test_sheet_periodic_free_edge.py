"""A periodic seam is an ordinary free edge when no metal joins it."""
import numpy as np
import pytest

from rfx.mesh_edges import solved_sheet_span


def assert_mirrored_free_edges(left, right):
    assert (left.free_lo, left.free_hi) == (True, True), 'free ends'
    assert (right.free_lo, right.free_hi) == (True, True), 'free ends'
    assert left.hi - left.lo == pytest.approx(right.hi - right.lo), 'mirror width'
    assert left.hi - left.lo == pytest.approx(3.7), 'fringe width'


def spans():
    nodes = np.arange(10, dtype=float)
    result = []
    for lo, hi in ((0, 3), (6, 9)):
        footprint = np.zeros((10, 5, 3), dtype=bool)
        footprint[lo:hi + 1, 1:4, 1] = True
        result.append(solved_sheet_span(footprint, 0, nodes, lo - .35, hi + .35,
                                       10., periodic=True, normal_axis=2))
    return result


def test_periodic_end_node_and_mirror_are_free():
    assert_mirrored_free_edges(*spans())


def test_free_edge_check_rejects_wall_guard():
    left, right = spans()
    with pytest.raises(AssertionError, match='free ends'):
        assert_mirrored_free_edges(left._replace(free_lo=False), right)
