"""Contract tests for the design-interop material codec.

Pure structural checks — no FDTD.  The existing scene artifact reduces
dispersion pole lists to ``{"present", "count"}``, so a material rebuilt from
it would silently lose its dispersion; these tests pin the lossless path.
"""

from __future__ import annotations

import dataclasses
import json

import pytest

from rfx.api._spec import MaterialSpec
from rfx.interop import UnsupportedDesignFeature
from rfx.interop._materials import (
    _POLE_FIELDS,
    _SCALAR_FIELDS,
    material_from_dict,
    material_to_dict,
    materials_from_dict,
    materials_to_dict,
)
from rfx.materials.debye import DebyePole
from rfx.materials.lorentz import LorentzPole


def _fr4() -> MaterialSpec:
    return MaterialSpec(eps_r=4.3, sigma=0.0, mu_r=1.0)


def _dispersive() -> MaterialSpec:
    return MaterialSpec(
        eps_r=2.1,
        sigma=1e-3,
        mu_r=1.2,
        chi3=3.5e-22,
        debye_poles=[DebyePole(delta_eps=1.4, tau=2.3e-12)],
        lorentz_poles=[
            LorentzPole(omega_0=2 * 3.141592653589793 * 6e9, delta=1e8, kappa=0.7),
            LorentzPole(omega_0=2 * 3.141592653589793 * 9e9, delta=2e8, kappa=0.3),
        ],
    )


def test_registry_pins_live_material_spec_fields():
    """A new MaterialSpec field must fail here, not vanish from exports."""
    live = {f.name for f in dataclasses.fields(MaterialSpec)}
    recorded = set(_SCALAR_FIELDS) | set(_POLE_FIELDS)
    assert live == recorded, (
        "MaterialSpec and the interop registry disagree; update "
        "rfx/interop/_materials.py"
    )


@pytest.mark.parametrize("kind", sorted(_POLE_FIELDS))
def test_registry_pins_live_pole_parameters(kind):
    pole_cls, names = _POLE_FIELDS[kind]
    assert tuple(pole_cls._fields) == names, (
        f"{pole_cls.__name__} parameters changed; update "
        f"rfx/interop/_materials.py"
    )


@pytest.mark.parametrize("factory", [_fr4, _dispersive])
def test_round_trip_through_json_text(factory):
    original = factory()
    rebuilt = material_from_dict(json.loads(json.dumps(material_to_dict(original))))
    assert rebuilt == original


def test_dispersion_poles_survive_with_parameters():
    """The gap this codec exists to close (rfx/artifacts.py:245)."""
    payload = material_to_dict(_dispersive())
    assert payload["debye_poles"] == [{"delta_eps": 1.4, "tau": 2.3e-12}]
    assert len(payload["lorentz_poles"]) == 2
    assert payload["lorentz_poles"][0]["kappa"] == pytest.approx(0.7)
    assert payload["chi3"] == pytest.approx(3.5e-22)


def test_absent_poles_stay_none_not_empty_list():
    """None and [] are different states; the codec must not conflate them."""
    payload = material_to_dict(_fr4())
    assert payload["debye_poles"] is None
    assert payload["lorentz_poles"] is None

    empty = MaterialSpec(eps_r=1.0, debye_poles=[])
    assert material_to_dict(empty)["debye_poles"] == []
    assert material_from_dict(material_to_dict(empty)).debye_poles == []


def test_mapping_round_trip():
    materials = {"fr4": _fr4(), "resin": _dispersive()}
    rebuilt = materials_from_dict(
        json.loads(json.dumps(materials_to_dict(materials))))
    assert rebuilt == materials


def test_non_material_spec_is_refused():
    class FakeMaterial:
        eps_r = 4.3
        sigma = 0.0
        mu_r = 1.0

    with pytest.raises(UnsupportedDesignFeature, match="expected MaterialSpec"):
        material_to_dict(FakeMaterial())


def test_foreign_pole_type_is_refused():
    bogus = MaterialSpec(eps_r=1.0, debye_poles=[(1.4, 2.3e-12)])
    with pytest.raises(UnsupportedDesignFeature, match="expected DebyePole"):
        material_to_dict(bogus)


def test_pole_parameter_mismatch_is_refused():
    payload = material_to_dict(_dispersive())
    del payload["debye_poles"][0]["tau"]
    with pytest.raises(UnsupportedDesignFeature, match="parameter mismatch"):
        material_from_dict(payload)


def test_missing_field_in_payload_is_refused():
    payload = material_to_dict(_fr4())
    del payload["mu_r"]
    with pytest.raises(UnsupportedDesignFeature, match="missing="):
        material_from_dict(payload)


def test_unknown_field_in_payload_is_refused():
    payload = material_to_dict(_fr4())
    payload["tan_delta"] = 0.02
    with pytest.raises(UnsupportedDesignFeature, match="unknown="):
        material_from_dict(payload)
