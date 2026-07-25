"""Contract tests pinning the published design-IR schema against the emitter.

A schema that disagrees with the code it documents is worse than no schema: a
reader validates against it, passes, and believes something false. These tests
fail if either side drifts.

Pure structural checks — no FDTD.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from rfx import Simulation
from rfx.api._spec import MaterialSpec
from rfx.geometry.csg import Box, Cylinder
from rfx.interop import design_to_dict, design_to_json
from rfx.interop._design import (
    DESIGN_SCHEMA_VERSION,
    _EXCITATION_KEYS,
    _OBSERVABLE_KEYS,
    _TOP_LEVEL_KEYS,
)
from rfx.interop._shapes import SUPPORTED_SHAPE_KINDS
from rfx.materials.debye import DebyePole

_REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = _REPO_ROOT / "docs/design_notes/schemas/rfx-design-ir-v1.schema.json"


@pytest.fixture(scope="module")
def schema() -> dict:
    return json.loads(SCHEMA_PATH.read_text())


@pytest.fixture(scope="module")
def documents() -> list[dict]:
    """A few progressively richer designs, exported."""
    docs = []

    plain = Simulation(freq_max=10e9, domain=(0.02, 0.012, 0.004), dx=2e-4,
                       boundary="cpml")
    plain.add_material("fr4", eps_r=4.3, sigma=0.0)
    plain.add(Box(corner_lo=(0.0, 0.0, 0.0),
                  corner_hi=(0.02, 0.012, 0.0015)), material="fr4")
    plain.add(Cylinder(center=(0.010, 0.006, 0.0008), radius=3e-4,
                       height=1.5e-3, axis="z"), material="pec")
    plain.add_probe(position=(0.010, 0.006, 0.0016), component="ez")
    docs.append(design_to_dict(plain))

    # smooth_grading is what rfx itself recommends for an abrupt profile, and it
    # inserts transition cells — so the profile is also a harder serialisation
    # case (uneven values, not three flat blocks).
    from rfx import smooth_grading
    dz = np.asarray(smooth_grading(
        np.concatenate([np.full(20, 2e-4), np.full(30, 5e-5), np.full(20, 2e-4)])))
    graded = Simulation(freq_max=12e9, domain=(0.02, 0.012, float(dz.sum())),
                        dx=2e-4, boundary="cpml", cpml_layers=12, dz_profile=dz)
    graded.add_material(
        "lossy", eps_r=3.5, sigma=0.02,
        debye_poles=[DebyePole(delta_eps=1.1, tau=1.7e-12)])
    graded.add(Box(corner_lo=(0.0, 0.0, 0.0),
                   corner_hi=(0.02, 0.012, 0.001)), material="lossy")
    docs.append(design_to_dict(graded))

    pec = Simulation(freq_max=8e9, domain=(0.03, 0.02, 0.01), dx=5e-4,
                     boundary="pec")
    pec.add(Box(corner_lo=(0.001, 0.001, 0.001),
                corner_hi=(0.004, 0.004, 0.004)), material="pec")
    docs.append(design_to_dict(pec))

    return docs


def test_schema_file_exists_and_declares_the_right_identity(schema):
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert schema["$id"].endswith("rfx-design-ir-v1.schema.json")
    assert schema["properties"]["schema"]["const"] == DESIGN_SCHEMA_VERSION


def test_schema_top_level_keys_match_the_emitter(schema):
    """If the emitter gains or loses a section, this fails rather than the
    schema silently documenting a stale shape."""
    assert set(schema["required"]) == _TOP_LEVEL_KEYS
    assert set(schema["properties"]) == _TOP_LEVEL_KEYS
    assert schema["additionalProperties"] is False, (
        "the document is a closed world; the schema must say so"
    )


def test_schema_excitation_keys_match_the_emitter(schema):
    section = schema["properties"]["excitations"]
    assert set(section["required"]) == _EXCITATION_KEYS
    assert set(section["properties"]) == _EXCITATION_KEYS
    assert section["additionalProperties"] is False


def test_schema_observable_keys_match_the_emitter(schema):
    section = schema["properties"]["observables"]
    assert set(section["required"]) == _OBSERVABLE_KEYS
    assert set(section["properties"]) == _OBSERVABLE_KEYS
    assert section["additionalProperties"] is False


def test_schema_shape_kinds_match_the_codec(schema):
    declared = schema["$defs"]["shape"]["properties"]["kind"]["enum"]
    assert sorted(declared) == sorted(SUPPORTED_SHAPE_KINDS)


def test_schema_shape_kinds_exclude_mesh_shape(schema):
    """MeshShape is refused, not degraded — the schema must not imply support."""
    declared = schema["$defs"]["shape"]["properties"]["kind"]["enum"]
    assert not any("mesh" in kind for kind in declared)


@pytest.mark.parametrize("index", [0, 1, 2])
def test_emitted_documents_validate_against_the_published_schema(
        schema, documents, index):
    jsonschema = pytest.importorskip("jsonschema")
    jsonschema.validate(instance=documents[index], schema=schema)


def test_documents_carry_no_derived_or_run_control_state(documents):
    """The schema forbids extra keys; this pins the specific things that must
    never appear, since those are the ones a reader would wrongly trust."""
    forbidden = ("dt", "grid_shape", "axis_pads", "n_steps", "until_decay",
                 "compute_s_params", "eps_override", "preflight", "result")
    for document in documents:
        text = json.dumps(document)
        for key in forbidden:
            assert f'"{key}"' not in text, f"{key} must not appear in a design document"


def test_json_output_rejects_non_standard_float_tokens(documents):
    """json.dumps would happily write NaN/Infinity, which no strict reader
    accepts; the emitter must not rely on that."""
    plain = Simulation(freq_max=10e9, domain=(0.02, 0.012, 0.004), dx=2e-4,
                       boundary="cpml")
    plain.add(Box(corner_lo=(0.0, 0.0, 0.0),
                  corner_hi=(0.02, 0.012, 0.0015)), material="pec")
    text = design_to_json(plain)
    assert "NaN" not in text and "Infinity" not in text
    json.loads(text)  # strict parse


def test_material_payload_shape_is_pinned(schema):
    """The material payload is the part most likely to drift silently, because
    artifacts.py already reduces poles to {present, count}."""
    payload = schema["properties"]["materials"]["additionalProperties"]
    assert set(payload["required"]) == {
        "eps_r", "sigma", "mu_r", "chi3", "debye_poles", "lorentz_poles"}

    live = {f.name for f in __import__("dataclasses").fields(MaterialSpec)}
    assert set(payload["required"]) == live, (
        "MaterialSpec and the published schema disagree"
    )

    debye = payload["properties"]["debye_poles"]["items"]
    lorentz = payload["properties"]["lorentz_poles"]["items"]
    assert set(debye["required"]) == set(DebyePole._fields)
    from rfx.materials.lorentz import LorentzPole
    assert set(lorentz["required"]) == set(LorentzPole._fields)
