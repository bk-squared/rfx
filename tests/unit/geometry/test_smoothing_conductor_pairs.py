"""Realized metal leaves the surrounding dielectric average unchanged (#1778)."""

import numpy as np
import pytest

from rfx import Box, Cylinder, Simulation
from rfx.geometry.smoothing import (
    compute_smoothed_eps,
    compute_smoothed_eps_nonuniform,
    smoothed_shape_pairs,
)


@pytest.mark.parametrize("graded", [False, True], ids=["uniform", "graded-z"])
@pytest.mark.parametrize("boundary", ["pec", "cpml"])
@pytest.mark.parametrize("metal", ["pec", "copper"])
def test_conductor_leaves_dielectric_edges_unchanged(graded, boundary, metal):
    dx, width, length = .5e-3, 8e-3, 24e-3
    profile = dx * np.array([1.] * 20 + [.75] + [.5] * 9 + [.75] + [1.] * 22)

    def scene(with_conductor):
        sim = Simulation(freq_max=10e9, domain=(width, width, length), dx=dx,
                         boundary=boundary, cpml_layers=2,
                         **({"dz_profile": profile} if graded else {}))
        sim.add_material("fill", eps_r=4.)
        sim.add(Box((0., 0., 0.), (width, width, length)), material="fill")
        if with_conductor:
            sim.add_material("copper", sigma=5.8e7)
            sim.add(Cylinder((3.7e-3, 4.2e-3, length / 2), radius=1e-3,
                             height=length, axis="z"), material=metal)
        return sim

    reference, conductor = scene(False), scene(True)
    grid = conductor._build_realized_grid()
    smooth = compute_smoothed_eps_nonuniform if graded else compute_smoothed_eps
    # Independent reference: the no-conductor declaration, without the pair builder.
    expected = smooth(grid, [(entry.shape, 4.) for entry in reference._geometry],
                      background_eps=1.)
    if boundary == "cpml":
        # Fill the pads independently of the continuation code under test.
        expected = smooth(grid, [(Box((-width, -width, -length),
                                     (2 * width, 2 * width, 2 * length)), 4.)],
                          background_eps=1.)
    pairs, findings = smoothed_shape_pairs(conductor, grid)
    actual = smooth(grid, pairs, background_eps=1.)
    assert findings == []
    # The public record previews the same realized Yee PEC edges as the runners.
    masks = conductor.realized_geometry().edge_masks
    assert len(masks) == 3 and any(mask.any() for mask in masks)
    for component, got, want, mask in zip("xyz", actual, expected, masks):
        outside = ~np.asarray(mask)
        assert outside.any()
        np.testing.assert_allclose(np.asarray(got)[outside], np.asarray(want)[outside],
                                   rtol=1e-6, atol=1e-6, err_msg=f"E{component}, {metal}")
