"""The battery solves the drawn board before either DUT takes a time step."""

import numpy as np

from tests import _chain_battery_drift as drift


def test_battery_board_is_solved_as_drawn():
    """Pin free sheet bounds and the whole seam independently of the driver guard."""
    driver = drift.load_driver("scripts/diagnostics/msl_chain_battery_measure.py")
    profiles = []
    for dut, ladder in (("notch", (27, 3)), ("thru", (18, 39))):
        sim = driver.build_sim(100e-6, dut)
        geo = driver.assert_realized(sim, 100e-6, dut)
        assert geo["snap"] == "strict"
        assert geo["lane"] == "nonuniform"
        np.testing.assert_allclose(geo["free_edges_m"]["trace_y"], [0.003, 0.0036],
                                   rtol=0, atol=1e-9)
        assert geo["substrate_cells_under_strip"] == 3
        for port, feed in zip(geo["ports"], (0.003, 0.023), strict=True):
            assert abs(port["feed_plane_m"] - feed) <= 1e-12
            assert (port["n_probe_offset"], port["n_probe_spacing"]) == ladder
        if dut == "notch":
            np.testing.assert_allclose(geo["free_edges_m"]["stub_x"], [0.0127, 0.0133],
                                       rtol=0, atol=1e-9)
            np.testing.assert_allclose(geo["free_edges_m"]["stub_open_y"], [0.0156],
                                       rtol=0, atol=1e-9)
            assert geo["seam"]["metal_nodes"] == geo["seam"]["stub_x_nodes"] == 7, geo["seam"]
            assert not geo["seam"]["outside_metal"]
            assert geo["seam"]["stub_y_free_ends"] == [False, True]
        else:
            assert geo["seam"] is None
        profiles.append(geo["profiles"])
    assert profiles[0] == profiles[1]
