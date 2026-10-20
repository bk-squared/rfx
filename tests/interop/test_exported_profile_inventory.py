"""The export inventory must not promise traced-only mesh declarations."""
import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Simulation
from rfx.interop._design import (
    EXCLUDED_SIMULATION_ATTRS, EXPORTED_SIMULATION_ATTRS,
    design_to_dict, design_to_json, simulation_from_design,
)
from rfx.interop._errors import UnsupportedDesignFeature


def test_nominal_profiles_are_excluded_from_export_inventory():
    assert "_nominal_profiles" not in EXPORTED_SIMULATION_ATTRS
    assert "_nominal_profiles" in EXCLUDED_SIMULATION_ATTRS
    sim = Simulation(freq_max=10e9, domain=(.008, .008, .008), dx=.001, boundary="pec")
    document = design_to_dict(sim)
    assert design_to_dict(simulation_from_design(document)) == document
    assert design_to_dict(simulation_from_design(json.loads(design_to_json(sim)))) == document

    def dump(delta):
        traced = Simulation(
            freq_max=10e9, domain=(.008, .008, .008), dx=.001, boundary="pec",
            dz_profile=jnp.full(8, .001) + delta,
            nominal_profiles={"z": np.full(8, .001)})
        design_to_dict(traced)
        return delta

    with pytest.raises(UnsupportedDesignFeature, match="trac"):
        jax.jit(dump)(jnp.float32(0.))
