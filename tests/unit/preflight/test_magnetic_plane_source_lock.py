"""A soft Ez source on a magnetic face excites its sheet but not the volume.

In a 16 × 12 × 6 mm box, the present half-cell magnetic wall leaves the
off-plane probe at exactly 0 V/m. This records the wall behaviour described
by the source advisory; moving the wall onto its face must revisit that text.
"""

import numpy as np

from rfx import Simulation
from rfx.boundaries.spec import BoundarySpec


def test_magnetic_plane_source_leaves_volume_exactly_zero():
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
    assert np.all(fields[:, 1] == 0.0), (
        "the magnetic-plane source advisory in rfx/preflight/sources.py (#1219) "
        "describes the half-cell wall; update it with the wall (#1221)"
    )
