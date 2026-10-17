"""The estimator refuses the same unsupported design-box lanes as forward."""
import jax.numpy as jnp
import numpy as np
import pytest

from rfx.api import Simulation


@pytest.mark.parametrize("lane", ("adi", "distributed", "graded-occupancy", "cpml"))
def test_estimator_and_forward_refuse_design_box(lane):
    options = dict(solver="adi", boundary="pec", cpml_layers=0) if lane == "adi" else dict(boundary="cpml", cpml_layers=2)
    if lane == "graded-occupancy":
        options["dz_profile"] = np.full(12, .001)
    sim = Simulation(freq_max=10e9, domain=(.012,) * 3, dx=.001, **options)
    sim.add_source((.002, .002, .002), "ez")
    sim.add_probe((.009, .009, .009), "ez")
    grid = sim._build_nonuniform_grid() if lane == "graded-occupancy" else sim._build_grid()
    box = ((-.001,) * 3, (.0005,) * 3) if lane == "cpml" else ((.004,) * 3, (.006,) * 3)
    bounds = sim._design_box_bounds_from_corners(grid, box)
    shape = tuple(bounds[2*a+1] - bounds[2*a] for a in range(3))
    occupancy = lane == "graded-occupancy"
    common = dict(design_box=box, checkpoint=False, distributed=lane == "distributed")
    error = ValueError if lane == "cpml" else NotImplementedError
    with pytest.raises(ValueError if lane == "distributed" else error):
        sim.estimate_ad_memory(4, design="occupancy" if occupancy else "permittivity", **common)
    with pytest.raises(error):
        sim.forward(n_steps=4, skip_preflight=True, **common,
                    **{"design_occupancy_override" if occupancy else "design_eps_override": jnp.ones(shape, jnp.float32)})
