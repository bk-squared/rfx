"""Tangential E on the magnetic E-node face couples into the volume.

The odd H image connects the face node to the interior (#1221 B3b).
"""

import numpy as np

from rfx import Simulation
from rfx.boundaries.spec import BoundarySpec


def test_magnetic_plane_source_couples_into_volume():
    sim = Simulation(
        freq_max=20e9, domain=(0.016, 0.012, 0.006), dx=1e-3,
        boundary=BoundarySpec(x="pmc", y="pec", z="pec"), cpml_layers=0,
    )
    sim.add_source((0.0, 0.006, 0.003), "ez", amplitude_kind="field")
    sim.add_probe((0.0, 0.009, 0.003), "ez")
    sim.add_probe((0.008, 0.006, 0.003), "ez")
    result = sim.run(n_steps=100, compute_s_params=False, skip_preflight=True)
    fields = np.asarray(result.time_series)
    assert fields.shape == (100, 2)
    assert np.isfinite(fields).all()
    assert np.max(np.abs(fields[:, 0])) > 0.0, "the on-plane probe must see the source"
    assert np.max(np.abs(fields[:, 1])) > 0.0, "the face source must reach the volume"
