from __future__ import annotations

import json
from pathlib import Path

import pytest

from rfx.experiments import ExperimentSpec, ExperimentSpecError, compile_experiment


FIXTURE = (
    Path(__file__).parent / "fixtures" / "experiments" / "patch_antenna_cpu_v1.json"
)


def _document() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_golden_v1_spec_is_canonical_and_digest_stable():
    spec = ExperimentSpec.from_dict(_document())
    reparsed = ExperimentSpec.from_json(spec.canonical_json())

    assert spec.schema_version == "rfx-experiment/v1"
    assert spec.execution.backend == "cpu"
    assert reparsed == spec
    assert reparsed.sha256 == spec.sha256
    assert len(spec.sha256) == 64


def test_v0_migration_renames_blocks_and_preserves_identity_fields():
    current = _document()
    legacy = {
        "schema_version": "rfx-experiment/v0",
        "name": current["name"],
        "patch_antenna": current["model"],
        "run": current["execution"],
        "metadata": current["metadata"],
    }

    migrated = ExperimentSpec.from_dict(legacy)

    assert migrated == ExperimentSpec.from_dict(current)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda doc: doc.pop("schema_version"), "schema_version"),
        (lambda doc: doc.update({"surprise": True}), "unknown field"),
        (
            lambda doc: doc["model"].update({"cell_size_m": "0.002"}),
            "must be a number",
        ),
        (lambda doc: doc["execution"].update({"backend": "gpu"}), "only permits"),
        (
            lambda doc: doc["execution"].update({"precision": "float64"}),
            "float32.*mixed",
        ),
        (
            lambda doc: doc["model"]["frequency_sweep"].update({"stop_hz": 1.0}),
            "must exceed",
        ),
        (
            lambda doc: doc["model"]["patch"].update({"size_m": [1.0, 1.0]}),
            "fit strictly",
        ),
    ],
)
def test_invalid_specs_fail_loudly(mutate, message):
    document = _document()
    mutate(document)
    with pytest.raises(ExperimentSpecError, match=message):
        ExperimentSpec.from_dict(document)


def test_compiler_is_deterministic_and_uses_native_config_builder():
    first = compile_experiment(_document())
    second = compile_experiment(_document())

    assert first.sha256 == second.sha256
    assert first.generated_python == second.generated_python
    assert first.config == second.config
    assert first.config["boundary"] == "cpml"
    assert first.config["execution"]["compute_s_params"] is True
    assert len(first.config["geometry"]) == 3
    assert len(first.config["sources"]) == 1
    assert "os.environ['JAX_PLATFORMS'] = 'cpu'" in first.generated_python

    namespace: dict = {}
    exec(compile(first.generated_python, "generated.py", "exec"), namespace)
    simulation = namespace["build_simulation"]()
    assert len(simulation._geometry) == 3
    assert len(simulation._ports) == 1
    assert (
        simulation._build_grid().shape == first.build_simulation()._build_grid().shape
    )


def test_compiled_preflight_is_a_structured_artifact():
    report = compile_experiment(_document()).preflight()

    assert set(report) == {"ok", "n_issues", "n_errors", "issues"}
    assert report["n_errors"] == 0
    assert report["ok"] is True
