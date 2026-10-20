"""The export inventory must not promise traced-only mesh declarations."""
import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Simulation
from rfx.interop._design import (
    EXCLUDED_SIMULATION_ATTRS, EXPORTED_SIMULATION_ATTRS, TRACED_ONLY_SIMULATION_ATTRS,
    design_to_dict, design_to_json, simulation_from_design,
)
from rfx.interop._errors import UnsupportedDesignFeature
from tests.interop.test_interop_design_document import _canonical
from tests.interop import test_interop_design_document as drift


@pytest.fixture(scope="module")
def documented_input():
    sim = Simulation(freq_max=10e9, domain=(.008, .008, .008), dx=.001, boundary="pec")
    document = design_to_dict(sim)
    return sim, document, simulation_from_design(document)


@pytest.mark.parametrize("attribute", EXPORTED_SIMULATION_ATTRS)
def test_every_listed_export_survives_documented_input(attribute, documented_input):
    original, _, rebuilt = documented_input
    assert _canonical(getattr(original, attribute)) == _canonical(getattr(rebuilt, attribute))


@pytest.mark.parametrize("function", [design_to_dict, design_to_json, simulation_from_design])
def test_documented_design_calls(function, documented_input):
    sim, document, _ = documented_input
    result = function(document if function is simulation_from_design else sim)
    if isinstance(result, str):
        result = json.loads(result)
    elif isinstance(result, Simulation):
        result = design_to_dict(result)
    assert result == document


def test_excluded_constructor_state_cannot_hide_from_drift_check(monkeypatch):
    def extra_state(**kwargs):
        sim = Simulation(**kwargs)
        sim._new_unserialized_state = 1
        return sim

    monkeypatch.setattr(drift, "Simulation", extra_state)
    monkeypatch.setattr(drift, "EXCLUDED_SIMULATION_ATTRS",
                        (*EXCLUDED_SIMULATION_ATTRS, "_new_unserialized_state"))
    with pytest.raises(AssertionError, match="constructor-state inventory has drifted"):
        drift.test_every_simulation_attribute_is_classified()


def test_nominal_profiles_are_excluded_from_export_inventory():
    assert "_nominal_profiles" not in EXPORTED_SIMULATION_ATTRS
    assert "_nominal_profiles" not in EXCLUDED_SIMULATION_ATTRS
    assert TRACED_ONLY_SIMULATION_ATTRS == ("_nominal_profiles",)
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
