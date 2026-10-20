"""Identify duplicates after positions resolve to component-owned cells."""
import numpy as np
import pytest

from rfx import Simulation
from rfx.ringdown import RingdownForward, RingdownRun, RingdownSpec


@pytest.mark.parametrize("lane", ["uniform", "graded"])
@pytest.mark.parametrize("planner", [RingdownRun, RingdownForward])
def test_duplicate_identification_probe_indices_warn_once(lane, planner):
    sim = Simulation(freq_max=10e9, domain=(.008, .008, .008), dx=.001,
                     boundary="pec",
                     **({"dz_profile": np.full(8, .001)} if lane == "graded" else {}))
    grid = sim._build_nonuniform_grid() if lane == "graded" else sim._build_grid()
    probes = (((.003, .003, .003), "ez"),
              ((.00301, .003, .003), "ez"),
              ((.003, .003, .003), "ex"),
              ((.003, .003, .003), "ez"))
    with pytest.warns(UserWarning, match=r"Duplicate.*indices \[\[0, 1, 3\]\]") as records:
        plan = planner(sim, RingdownSpec(identification_probes=probes),
                       lane=lane, n_steps=20, grid=grid)
    assert len(records) == 1
    assert len(plan.identification_keys) == 4
    assert plan.identification_keys[0] == plan.identification_keys[1]
    assert plan.identification_keys[0] != plan.identification_keys[2]
