"""The metric arrays of a graded z profile against an independent float64 calculation.

On a non-uniform profile the H curl uses the LOCAL cell length ``1/d[k]`` and the
E curl uses the DUAL edge length ``2/(d[k-1] + d[k])``. Get either wrong and every
resonance on a graded mesh moves; the pre-CORE-C2 defect was exactly the two
swapped. ``rfx.nonuniform._profile_to_inv_arrays`` builds those arrays for the
solver; ``tests/_nu_cavity_gates.py::inv_arrays`` is a float64 calculation that
shares no code with it.

This comparison used to live inside the cv24 cross-validation tests, removed on
2026-09-21, and was the only place the arrays themselves were checked (the
analytic cavity tests would notice a wrong metric only through a frequency).
"""
import numpy as np
import pytest

from rfx.nonuniform import _append_bounding_node, _profile_to_inv_arrays
from tests import _nu_cavity_gates as G


def _rfx_arrays(profile):
    inv_e, inv_h = _profile_to_inv_arrays(_append_bounding_node(profile))
    return np.asarray(inv_e, np.float64), np.asarray(inv_h, np.float64)


@pytest.mark.parametrize("name", ["single_band", "multi_band"])
def test_rfx_metric_arrays_are_dual_edge_for_e_and_local_cell_for_h(name):
    profile = G.PROFILES[name]
    want_e, want_h = G.inv_arrays(profile)
    got_e, got_h = _rfx_arrays(profile)
    # rfx stores float32, hence 2e-7 rather than machine precision.
    np.testing.assert_allclose(got_e, want_e, rtol=2e-7)
    np.testing.assert_allclose(got_h, want_h, rtol=2e-7)


def test_the_swapped_metric_defect_would_not_pass():
    """Negative control: with E given the local width and H the mean spacing (the
    pre-CORE-C2 defect) the arrays differ from rfx's by far more than 2e-7 at the
    graded cells, so the comparison above is not satisfied by both conventions."""
    profile = G.PROFILES["multi_band"]
    swap_e, swap_h = G.swapped_inv_arrays(profile)
    got_e, got_h = _rfx_arrays(profile)
    assert np.max(np.abs(got_e - swap_e) / np.abs(got_e)) > 1e-2
    assert np.max(np.abs(got_h[:-1] - swap_h[:-1]) / np.abs(got_h[:-1])) > 1e-2
