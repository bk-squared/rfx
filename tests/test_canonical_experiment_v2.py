from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import random
import subprocess
import sys

import pytest

from rfx.experiments import (
    CANONICAL_SCHEMA_VERSION,
    CanonicalExperimentSpec,
    ExperimentCompileError,
    build_scene_preview,
    compile_canonical_experiment,
    compile_experiment,
    simulation_semantic_fingerprint,
)


ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = ROOT / "tests" / "fixtures" / "experiments"
V2_FIXTURES = tuple(sorted(FIXTURE_ROOT.glob("*_v2.json")))


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _shuffle_object_keys(value, generator: random.Random):
    if isinstance(value, dict):
        items = list(value.items())
        generator.shuffle(items)
        return {key: _shuffle_object_keys(item, generator) for key, item in items}
    if isinstance(value, list):
        return [_shuffle_object_keys(item, generator) for item in value]
    return value


def test_three_golden_v2_specs_round_trip_and_have_distinct_workflows():
    specs = [CanonicalExperimentSpec.from_dict(_load(path)) for path in V2_FIXTURES]

    assert len(specs) == 3
    assert {spec.workflow for spec in specs} == {
        "patch_antenna",
        "wr90_waveguide",
        "multilayer_fresnel",
    }
    for spec in specs:
        assert spec.to_dict()["schema_version"] == CANONICAL_SCHEMA_VERSION
        assert CanonicalExperimentSpec.from_json(spec.canonical_json()) == spec
        assert len(spec.sha256) == 64
        assert len(spec.semantic_fingerprint) == 64


@pytest.mark.parametrize("path", V2_FIXTURES, ids=lambda path: path.stem)
def test_canonicalization_property_is_key_order_independent(path):
    original = CanonicalExperimentSpec.from_dict(_load(path))
    for seed in range(12):
        permuted = _shuffle_object_keys(_load(path), random.Random(seed))
        candidate = CanonicalExperimentSpec.from_dict(permuted)
        assert candidate == original
        assert candidate.canonical_json() == original.canonical_json()
        assert candidate.sha256 == original.sha256
        assert candidate.semantic_fingerprint == original.semantic_fingerprint


@pytest.mark.parametrize("path", V2_FIXTURES, ids=lambda path: path.stem)
def test_spec_simulation_generated_code_and_preview_share_fingerprint(path):
    compiled = compile_canonical_experiment(_load(path))
    simulation = compiled.build_simulation()
    preview = compiled.scene_preview()
    namespace: dict = {}
    exec(compile(compiled.generated_python, "generated.py", "exec"), namespace)

    assert compiled.semantic_fingerprint == compiled.spec.semantic_fingerprint
    assert simulation_semantic_fingerprint(simulation) == compiled.semantic_fingerprint
    assert preview["semantic_fingerprint"] == compiled.semantic_fingerprint
    assert namespace["SEMANTIC_FINGERPRINT"] == compiled.semantic_fingerprint
    assert (
        namespace["compiled_experiment"]().semantic_fingerprint
        == compiled.semantic_fingerprint
    )
    assert preview["entities"] or compiled.spec.workflow == "wr90_waveguide"
    assert preview["overlays"]


@pytest.mark.parametrize("path", V2_FIXTURES, ids=lambda path: path.stem)
def test_generated_code_fresh_process_matches_structured_preflight(path, tmp_path):
    compiled = compile_canonical_experiment(_load(path))
    generated = tmp_path / f"{path.stem}_generated.py"
    generated.write_text(compiled.generated_python, encoding="utf-8")
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT), env.get("PYTHONPATH", "")]).rstrip(
        os.pathsep
    )

    completed = subprocess.run(
        [sys.executable, str(generated)],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert json.loads(completed.stdout) == compiled.preflight()


def test_v1_patch_migrates_through_registry_to_v2():
    legacy = _load(FIXTURE_ROOT / "patch_antenna_cpu_v1.json")

    migrated = CanonicalExperimentSpec.from_dict(legacy)
    compiled = compile_canonical_experiment(migrated)

    assert migrated.workflow == "patch_antenna"
    assert migrated.to_dict()["schema_version"] == CANONICAL_SCHEMA_VERSION
    assert compiled.preflight()["ok"] is True


def test_public_compiler_dispatches_v2_and_json_schema_is_checked_in():
    document = _load(FIXTURE_ROOT / "patch_antenna_v2.json")
    compiled = compile_experiment(document)
    schema = json.loads(
        (
            ROOT / "docs" / "design_notes" / "schemas" / "rfx-experiment-v2.schema.json"
        ).read_text(encoding="utf-8")
    )

    assert compiled.spec.workflow == "patch_antenna"
    assert schema["properties"]["schema_version"]["const"] == CANONICAL_SCHEMA_VERSION
    assert set(schema["properties"]["kind"]["enum"]) == {
        "patch_antenna",
        "wr90_waveguide",
        "multilayer_fresnel",
    }


@pytest.mark.parametrize(
    ("mutation", "code", "path"),
    [
        (
            lambda document: document["geometry"][0].update({"kind": "sphere"}),
            "unsupported_variant",
            "$.geometry[0].kind",
        ),
        (
            lambda document: document["simulation"].update({"device": "gpu"}),
            "unknown_field",
            "$.simulation",
        ),
        (
            lambda document: document.update({"kind": "coaxial_network"}),
            "unsupported_workflow",
            "$.kind",
        ),
        (
            lambda document: document["observations"][1].update({"axis": "q"}),
            "unsupported_variant",
            "$.observations[1].axis",
        ),
        (
            lambda document: document["observations"][1].update(
                {"coordinate_m": 1.0}
            ),
            "range_error",
            "$.observations[1].coordinate_m",
        ),
    ],
)
def test_unsupported_or_unknown_features_return_coded_compile_errors(
    mutation, code, path
):
    document = _load(FIXTURE_ROOT / "patch_antenna_v2.json")
    mutation(document)

    with pytest.raises(ExperimentCompileError) as caught:
        compile_canonical_experiment(document)

    diagnostic = caught.value.diagnostics[0]
    assert diagnostic.code == code
    assert diagnostic.path == path


def test_metadata_edits_preserve_semantics_but_physics_edits_change_it():
    original = _load(FIXTURE_ROOT / "patch_antenna_v2.json")
    retitled = copy.deepcopy(original)
    retitled["metadata"]["title"] = "Retitled by a UI or agent"
    resized = copy.deepcopy(original)
    resized["geometry"][2]["bounds_m"][1][0] += 0.001

    base = CanonicalExperimentSpec.from_dict(original)
    title_only = CanonicalExperimentSpec.from_dict(retitled)
    physics_change = CanonicalExperimentSpec.from_dict(resized)

    assert base.sha256 != title_only.sha256
    assert base.semantic_fingerprint == title_only.semantic_fingerprint
    assert base.semantic_fingerprint != physics_change.semantic_fingerprint


def test_2d_field_snapshot_off_plane_request_warns_and_compiles_on_solved_plane():
    document = _load(FIXTURE_ROOT / "multilayer_fresnel_v2.json")
    assert document["observations"][2]["coordinate_m"] == 0.0

    document["observations"][2]["coordinate_m"] = 0.02
    compiled = compile_canonical_experiment(document)

    diagnostic = next(
        item for item in compiled.diagnostics if item.code == "field_plane_snapped"
    )
    assert diagnostic.severity == "warning"
    assert diagnostic.path == "$.observations[2].coordinate_m"
    assert compiled.spec.to_dict()["observations"][2]["coordinate_m"] == 0.02
    assert compiled.config["observations"][2]["coordinate_m"] == 0.0


def test_scene_preview_is_spec_derived_and_does_not_require_solver(monkeypatch):
    import rfx.experiments.canonical as canonical

    document = _load(FIXTURE_ROOT / "patch_antenna_v2.json")
    monkeypatch.setattr(
        canonical,
        "_build_simulation",
        lambda _spec: (_ for _ in ()).throw(AssertionError("solver touched")),
    )

    preview = build_scene_preview(document)

    assert preview["schema_version"] == "rfx-scene-preview/v1"
    assert {entity["id"] for entity in preview["entities"]} == {
        "ground",
        "substrate",
        "patch",
    }
