import json
import numpy as np
import pytest

from rfx import Box, MeshPlan, Simulation, auto_configure, plan_mesh, plan_simulation_mesh


def _geometry():
    return [(Box((0.0, 0.0, 0.0), (0.02, 0.01, 0.002)), "fr4")]


def _materials():
    return {"fr4": {"eps_r": 4.0, "sigma": 0.0}}


def _schema_keys(plan):
    payload = plan.to_dict()
    return {
        "top": set(payload),
        "cell_sizes": set(payload["cell_sizes"]),
        "memory": set(payload["memory"]),
    }


EXPECTED_TOP_KEYS = {
    "schema_version",
    "plan_source",
    "freq_range",
    "accuracy",
    "boundary",
    "cell_sizes",
    "grid_shape",
    "domain",
    "margin",
    "cfl",
    "absorber",
    "resolution_basis",
    "support_checks",
    "memory",
    "artifact_declarations",
    "warnings",
    "recommendation",
}
EXPECTED_CELL_SIZE_KEYS = {
    "nominal_dx",
    "dx",
    "dy",
    "dx_min",
    "dx_max",
    "dy_min",
    "dy_max",
    "dz_min",
    "dz_max",
    "profiles_present",
}
EXPECTED_MEMORY_KEYS = {
    "source",
    "cells",
    "uniform_fine_cells",
    "cell_savings_factor",
    "estimated_mb",
    "max_memory_mb",
    "ad_memory",
    "status",
}


def test_plan_mesh_reuses_auto_config_values():
    geometry = _geometry()
    materials = _materials()

    cfg = auto_configure(geometry, (1.0e9, 3.0e9), materials=materials, accuracy="draft")
    plan = plan_mesh(geometry, (1.0e9, 3.0e9), materials=materials, accuracy="draft")

    assert isinstance(plan, MeshPlan)
    assert plan.schema_version == "mesh-plan/v1"
    assert plan.grid_shape == cfg.grid_shape
    assert plan.domain == cfg.domain
    assert plan.cell_sizes["dx"] == cfg.dx
    assert plan.cfl["dt"] == cfg.dt
    assert plan.to_sim_config().grid_shape == cfg.grid_shape
    sim_kwargs = plan.to_sim_kwargs()
    cfg_kwargs = cfg.to_sim_kwargs()
    assert {k: v for k, v in sim_kwargs.items() if k != "dz_profile"} == {
        k: v for k, v in cfg_kwargs.items() if k != "dz_profile"
    }
    if "dz_profile" in cfg_kwargs:
        assert sim_kwargs["dz_profile"].tolist() == cfg_kwargs["dz_profile"].tolist()
    assert any(row["source"] == "auto_configure" for row in plan.resolution_basis)
    assert plan.plan_source == "auto_configure"
    assert plan.cell_sizes["nominal_dx"] == cfg.dx
    assert plan.cell_sizes["dx_min"] == cfg.dx
    assert plan.cell_sizes["dy_min"] == cfg.dx
    assert "clearance_checks" not in plan.to_dict()


def test_plan_mesh_json_markdown_and_artifact_declarations(tmp_path):
    plan = plan_mesh(
        _geometry(),
        (1.0e9, 3.0e9),
        materials=_materials(),
        accuracy="draft",
        artifact_root=tmp_path / "artifacts",
    )

    payload = json.loads(plan.to_json())
    assert payload["artifact_declarations"]["mesh_plan"]["path"].endswith("mesh_plan.json")
    assert payload["artifact_declarations"]["report"]["path"].endswith("report.md")
    assert payload["artifact_declarations"]["scene"] == {"path": None, "status": "not_claimed"}
    assert payload["artifact_declarations"]["replay"] == {"path": None, "status": "not_claimed"}
    assert "# Mesh Plan" in plan.to_markdown()
    assert "not_claimed" in plan.to_markdown()
    assert not (tmp_path / "artifacts").exists()


def test_plan_simulation_mesh_suppresses_preflight_output_and_captures_sparam_issue(capsys):
    sim = Simulation(freq_max=3.0e9, domain=(0.03, 0.02, 0.01), boundary="pec", dx=0.005)

    plan = plan_simulation_mesh(sim, n_steps=4, sparameter_calculator="msl")

    captured = capsys.readouterr()
    assert captured.out == ""
    assert plan.accuracy is None
    assert plan.plan_source == "configured_simulation"
    assert plan.grid_shape == sim.mesh_intelligence_report().grid_shape
    assert any(row["source"] == "general_preflight" for row in plan.support_checks)
    assert any(
        row["source"] == "sparameter_preflight" and "No MSL ports" in row["message"]
        for row in plan.support_checks
    )
    payload = plan.to_dict()
    assert payload["freq_range"] == [None, 3.0e9]
    assert payload["margin"] is None
    assert "clearance_checks" not in payload
    assert payload["warnings"] == []
    assert payload["memory"]["estimated_mb"] is None
    assert payload["memory"]["ad_memory"] is not None
    assert all(row["message"] not in payload["warnings"] for row in payload["support_checks"])


def test_plan_simulation_mesh_propagates_unexpected_sparam_errors(monkeypatch):
    sim = Simulation(freq_max=3.0e9, domain=(0.03, 0.02, 0.01), boundary="pec", dx=0.005)

    def broken_preflight_sparameters(*args, **kwargs):
        raise RuntimeError("unexpected defect")

    monkeypatch.setattr(sim, "preflight_sparameters", broken_preflight_sparameters)

    with pytest.raises(RuntimeError, match="unexpected defect"):
        plan_simulation_mesh(sim, n_steps=4, sparameter_calculator="msl")




def test_plan_mesh_and_simulation_plan_share_schema_keys():
    geometry_plan = plan_mesh(
        _geometry(),
        (1.0e9, 3.0e9),
        materials=_materials(),
        accuracy="draft",
    )
    sim = Simulation(freq_max=3.0e9, domain=(0.03, 0.02, 0.01), boundary="pec", dx=0.005)
    simulation_plan = plan_simulation_mesh(sim, n_steps=4)

    assert _schema_keys(geometry_plan) == {
        "top": EXPECTED_TOP_KEYS,
        "cell_sizes": EXPECTED_CELL_SIZE_KEYS,
        "memory": EXPECTED_MEMORY_KEYS,
    }
    assert _schema_keys(simulation_plan) == {
        "top": EXPECTED_TOP_KEYS,
        "cell_sizes": EXPECTED_CELL_SIZE_KEYS,
        "memory": EXPECTED_MEMORY_KEYS,
    }
    assert geometry_plan.to_dict()["freq_range"] == [1.0e9, 3.0e9]
    assert simulation_plan.to_dict()["freq_range"] == [None, 3.0e9]
    assert geometry_plan.margin is not None
    assert simulation_plan.margin is None
    assert simulation_plan.warnings == ()


def test_plan_simulation_mesh_reports_nonuniform_extrema_and_limiting_axis():
    sim = Simulation(
        freq_max=3.0e9,
        domain=(0.0, 0.0, 0.01),
        boundary="pec",
        dx=0.005,
        dx_profile=np.array([0.001, 0.002, 0.003]),
        dy_profile=np.array([0.002, 0.004]),
        dz_profile=np.array([0.0015, 0.0025, 0.0035]),
    )

    plan = plan_simulation_mesh(sim, n_steps=4)

    assert plan.cell_sizes["nominal_dx"] == 0.005
    assert plan.cell_sizes["dx"] == 0.005
    assert plan.cell_sizes["dy"] == 0.005
    assert plan.cell_sizes["dx_min"] == 0.001
    assert plan.cell_sizes["dx_max"] == 0.003
    assert plan.cell_sizes["dy_min"] == 0.002
    assert plan.cell_sizes["dy_max"] == 0.004
    assert plan.cell_sizes["dz_min"] == 0.0015
    assert plan.cell_sizes["dz_max"] == 0.0035
    assert plan.cell_sizes["profiles_present"] == {"x": True, "y": True, "z": True}
    assert plan.cfl["limiting_axis"] == "x"
def test_simulation_plan_mesh_method_matches_function():
    sim = Simulation(freq_max=3.0e9, domain=(0.03, 0.02, 0.01), boundary="pec", dx=0.005)

    method_plan = sim.plan_mesh(n_steps=4)
    function_plan = plan_simulation_mesh(sim, n_steps=4)

    assert method_plan.to_dict() == function_plan.to_dict()
    assert method_plan.memory["source"] == "Simulation.mesh_intelligence_report"
