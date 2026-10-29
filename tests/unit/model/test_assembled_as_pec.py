"""Material-resolution failures in the realized PEC predicate."""

from types import SimpleNamespace

import pytest

from rfx.model.materials import assembled_as_pec


@pytest.mark.parametrize("name", ["pec", "copper_x"])
def test_resolution_error_propagates(name):
    error = RuntimeError("forced material-resolution failure")

    def resolve(material_name):
        raise error

    sim = SimpleNamespace(_resolve_material=resolve)
    with pytest.raises(RuntimeError) as caught:
        assembled_as_pec(sim, SimpleNamespace(material_name=name))
    assert caught.value is error


@pytest.mark.parametrize("name,expected", [("pec", True), ("copper_x", False)])
def test_unknown_name_fallback(name, expected):
    def resolve(material_name):
        raise KeyError(material_name)

    sim = SimpleNamespace(_resolve_material=resolve)
    assert assembled_as_pec(sim, SimpleNamespace(material_name=name)) is expected
