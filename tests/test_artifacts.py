"""Runtime artifact/report/bundle export tests."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from rfx.api import Simulation
from rfx.geometry.csg import Box
from rfx.artifacts import (
    ALLOWED_REPRODUCIBILITY_STATUS,
    build_runtime_report,
    build_scene_artifact,
    export_artifact_bundle,
    render_artifact_markdown,
    validate_artifact_report,
)


def _minimal_sim(*, nonuniform: bool = False) -> Simulation:
    if nonuniform:
        dz = np.array([0.7e-3, 0.8e-3, 0.9e-3, 1.0e-3], dtype=np.float64)
        sim = Simulation(
            freq_max=8e9,
            domain=(0.008, 0.006, float(np.sum(dz))),
            boundary="pec",
            dx=1.0e-3,
            dz_profile=dz,
        )
    else:
        sim = Simulation(
            freq_max=8e9,
            domain=(0.008, 0.006, 0.004),
            boundary="pec",
            dx=1.0e-3,
        )
    sim.add_material("substrate", eps_r=3.55, sigma=0.0027)
    sim.add(Box((0.001, 0.001, 0.0005), (0.004, 0.004, 0.0015)), material="substrate")
    sim.add_port((0.002, 0.003, 0.001), "ez", impedance=50.0)
    sim.add_probe((0.006, 0.003, 0.001), "ez")
    return sim


def _fake_result(sim: Simulation) -> SimpleNamespace:
    grid = sim._build_grid()
    return SimpleNamespace(
        grid=grid,
        time_series=np.array([[0.0, 1.0], [0.5, -0.25]], dtype=np.float32),
        s_params=np.ones((1, 1, 3), dtype=np.complex64) * (0.5 + 0.1j),
        freqs=np.array([1.0e9, 2.0e9, 3.0e9]),
        snapshots={"ez": np.zeros((2, 3, 4), dtype=np.float32)},
        state=None,
        dt=float(grid.dt),
        freq_range=(1.0e9, 3.0e9),
    )


def test_build_scene_artifact_minimal_simulation():
    sim = _minimal_sim()
    scene = build_scene_artifact(sim)

    assert scene["freq_max"] == 8e9
    assert scene["domain"] == [0.008, 0.006, 0.004]
    assert scene["materials"]["substrate"]["eps_r"] == 3.55
    assert scene["geometry"][0]["shape_type"] == "Box"
    assert scene["geometry"][0]["material_name"] == "substrate"
    assert scene["geometry"][0]["bounding_box"] == [
        [0.001, 0.001, 0.0005],
        [0.004, 0.004, 0.0015],
    ]
    assert "values" not in json.dumps(scene).lower()


def test_export_scene_method_returns_dict_and_writes_json(tmp_path: Path):
    sim = _minimal_sim()
    scene = sim.export_scene()
    path = sim.export_scene(tmp_path / "scene.json")

    assert isinstance(scene, dict)
    assert path == tmp_path / "scene.json"
    written = json.loads(path.read_text())
    assert written["geometry"][0]["shape_type"] == scene["geometry"][0]["shape_type"]


def test_build_runtime_report_without_result():
    report = build_runtime_report(_minimal_sim())

    for key in (
        "simulation", "scene", "cad_compat", "mesh", "preflight", "result",
        "visualization", "bundle", "provenance", "reproducibility", "limitations",
    ):
        assert key in report
    assert report["cad_compat"]["source_type"] == "native-rfx-scene"
    assert report["cad_compat"]["status"] == "not-cad-import"
    assert report["reproducibility"]["status"] in ALLOWED_REPRODUCIBILITY_STATUS
    assert report["reproducibility"]["status"] != "replayable"
    assert validate_artifact_report(report) == []


def test_build_runtime_report_with_result_like_object():
    sim = _minimal_sim()
    report = build_runtime_report(sim, _fake_result(sim))
    result = report["result"]

    assert result["status"] == "provided"
    assert result["grid"]["shape"] == list(sim._build_grid().shape)
    assert result["time_series"]["shape"] == [2, 2]
    assert result["time_series"]["peak"] == 1.0
    assert result["s_params"]["shape"] == [1, 1, 3]
    assert result["freqs"]["count"] == 3
    assert result["snapshots"]["ez"]["shape"] == [2, 3, 4]
    # Large result arrays are summarized by metadata, not serialized as values.
    assert "[[0.0, 1.0]" not in json.dumps(result)


def test_render_artifact_markdown_core_sections():
    report = build_runtime_report(_minimal_sim())
    md = render_artifact_markdown(report)

    for heading in (
        "## Simulation", "## Native Scene / CAD Bridge", "## Mesh / Preflight",
        "## Result", "## Visualization", "## Bundle", "## Limitations",
    ):
        assert heading in md
    assert "No CAD import is performed" in md
    assert "Deterministic replay is not claimed" in md
    assert "GUI support" not in md


def test_validate_artifact_report_required_top_level_and_nested_fields():
    report = build_runtime_report(_minimal_sim())

    for field in [
        "simulation", "scene", "cad_compat", "mesh", "preflight", "result",
        "visualization", "bundle", "provenance", "reproducibility", "limitations",
    ]:
        mutated = copy.deepcopy(report)
        mutated.pop(field)
        assert validate_artifact_report(mutated), field

    for section, fields in {
        "simulation": ("freq_max", "domain", "mode", "boundary", "cpml_layers", "dx", "mesh_profiles"),
        "scene": ("materials", "geometry", "sources", "ports", "probes"),
        "cad_compat": ("source_type", "status", "entities", "limitations"),
        "mesh": ("status", "grid_shape", "cell_count", "grid_type", "nonuniform", "audit"),
        "preflight": ("status", "issues", "warnings_source"),
        "result": ("status", "result_type", "grid", "time_series", "s_params", "freqs", "snapshots"),
        "visualization": ("outputs", "primary_view", "checks"),
        "bundle": ("files", "manifest_path", "validation_status"),
        "provenance": ("python", "platform", "rfx_version", "cwd", "command"),
        "reproducibility": ("status", "repository", "commit", "worktree_status", "command", "inputs", "limitations"),
    }.items():
        for field in fields:
            mutated = copy.deepcopy(report)
            mutated[section].pop(field)
            assert validate_artifact_report(mutated), f"{section}.{field}"


def test_validate_artifact_report_rejects_imported_cad_claim():
    report = build_runtime_report(_minimal_sim())
    report["cad_compat"]["status"] = "imported"
    report["cad_compat"]["source_type"] = "step"

    errors = validate_artifact_report(report)
    assert any("CAD import" in error or "cad_compat" in error for error in errors)


def test_reproducibility_section_is_provenance_not_replay_claim():
    report = build_runtime_report(_minimal_sim())
    repro = report["reproducibility"]

    assert {"repository", "commit", "worktree_status", "command", "inputs"} <= set(repro)
    assert isinstance(repro["inputs"], list)
    report["reproducibility"]["status"] = "replayable"
    assert validate_artifact_report(report)
    md = render_artifact_markdown(report).lower()
    assert "deterministic replay is not claimed" in md
    assert "deterministic replay is claimed" not in md


def test_export_artifact_bundle_writes_manifest_and_hashes(tmp_path: Path):
    bundle = export_artifact_bundle(tmp_path / "bundle", _minimal_sim())

    for path in (bundle.report_json, bundle.report_markdown, bundle.scene_json, bundle.geometry_json, bundle.manifest_json):
        assert path is not None
        assert path.exists()

    report = json.loads(bundle.report_json.read_text())
    assert validate_artifact_report(report, bundle_root=bundle.root) == []
    manifest = json.loads(bundle.manifest_json.read_text())
    assert manifest["files"]
    for entry in manifest["files"]:
        rel = Path(entry["path"])
        assert not rel.is_absolute()
        data = (bundle.root / rel).read_bytes()
        assert hashlib.sha256(data).hexdigest() == entry["sha256"]


def test_bundle_validation_detects_missing_declared_file(tmp_path: Path):
    bundle = export_artifact_bundle(tmp_path / "bundle", _minimal_sim())
    report = json.loads(bundle.report_json.read_text())
    assert bundle.scene_json is not None
    bundle.scene_json.unlink()

    errors = validate_artifact_report(report, bundle_root=bundle.root)
    assert any("scene.json" in error for error in errors)


def test_field_vtk_is_opt_in_and_graceful_without_inputs(tmp_path: Path):
    default = export_artifact_bundle(tmp_path / "default", _minimal_sim())
    assert default.field_vtk is None

    bundle = export_artifact_bundle(tmp_path / "vtk", _minimal_sim(), include_field_vtk=True)
    report = json.loads(bundle.report_json.read_text())
    assert bundle.field_vtk is None
    assert any(
        item.get("role") == "field-vtk" and item.get("status") == "missing-input"
        for item in report["visualization"]["outputs"]
    )


def test_nonuniform_mesh_report_included_when_profiles_present():
    report = build_runtime_report(_minimal_sim(nonuniform=True))

    assert report["mesh"]["nonuniform"] is True
    assert report["mesh"]["profiles"]["dz_profile"]["present"] is True
    assert report["mesh"]["grid_shape"] is not None
    assert report["mesh"]["audit"]["status"] in {"available", "unavailable"}


def test_preflight_capture_is_structured():
    report = build_runtime_report(_minimal_sim())
    preflight = report["preflight"]

    assert preflight["status"] in {"passed", "issues", "error"}
    assert isinstance(preflight["issues"], list)
    assert "warnings_source" in preflight
