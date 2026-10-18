"""#1881: declared 1 mm widths and the same current product on both meshes.

The matrix uses binary-exact 1/1024 m widths. This issue scene uses 1 mm.
The far probe first differs at step 11 with the float32-store volume; the
source-cell probe first differs at step 4 with the (Cb*(1/dV))*w product.
"""
import numpy as np
import pytest

from rfx import Simulation
from rfx.boundaries.spec import Boundary, BoundarySpec


@pytest.mark.parametrize('floor', ('pec', 'pmc'))
def test_current_constant_profile_is_bit_identical(floor):
    d = 1e-3
    series = []
    for graded in (False, True):
        sim = Simulation(freq_max=10e9, domain=(16*d, 12*d, 8*d), dx=d, cpml_layers=8,
                         boundary=BoundarySpec(x='cpml', y='pec', z=Boundary(floor, 'pec')),
                         **({'dz_profile': np.full(8, d)} if graded else {}))
        sim.add_source((4*d, 6*d, 3.5*d), 'ez', amplitude_kind='current')
        sim.add_probe((10*d, 6*d, .5*d), 'ez')
        sim.add_probe((4*d, 6*d, 3.5*d), 'ez')
        series.append(np.asarray(sim.run(n_steps=11, compute_s_params=False).time_series))
    assert np.max(np.abs(series[0])) > 0
    np.testing.assert_array_equal(series[0].view(np.uint32), series[1].view(np.uint32))
