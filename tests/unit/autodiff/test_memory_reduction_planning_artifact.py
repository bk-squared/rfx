from __future__ import annotations

import json

from scripts.memory_reduction_planning_artifact import build_artifact, main


def test_memory_reduction_artifact_covers_budget_and_cell_savings():
    artifact = build_artifact(n_steps=5_000, available_memory_gb=1.0)

    assert artifact["status"] == "memory_reduction_planning_ready"
    assert all(artifact["gates"].values())
    assert artifact["plan"]["checkpoint_every"] is not None
    assert artifact["plan"]["segmented_fits"] is True
    assert artifact["mesh_report"]["cell_savings_factor"] >= 40.0
    assert artifact["mesh_report"]["preflight_issues"] == []
    assert artifact["mesh_report"]["ad_memory"]["ad_segmented_gb"] < artifact["mesh_report"]["ad_memory"]["ad_full_gb"]


def test_memory_reduction_artifact_cli_writes_json(tmp_path):
    output = tmp_path / "memory_plan.json"

    assert main(["--available-memory-gb", "1.0", "--output", str(output)]) == 0
    artifact = json.loads(output.read_text())

    assert artifact["gates"]["segmented_ad_fits_budget"] is True
    assert artifact["plan"]["selected_estimate"]["ad_segmented_gb"] <= artifact["plan"]["target_memory_gb"]
