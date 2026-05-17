"""Stage 1: consolidated mesh-intelligence report.

The report should expose the practical non-uniform memory story without
replacing the existing preflight or AD-memory estimators.
"""

from __future__ import annotations

import json

import numpy as np

from rfx import MeshIntelligenceReport, Simulation, Box


def _patch_like_nonuniform_sim():
    ext = 40e-3
    dx = 0.5e-3
    dz = np.concatenate([np.full(20, 0.3e-3), np.full(30, 0.6e-3)])
    sim = Simulation(
        freq_max=10e9,
        domain=(ext, ext, float(np.sum(dz))),
        dx=dx,
        dz_profile=dz,
        boundary="cpml",
        cpml_layers=8,
    )
    sim.add_source((ext / 2, ext / 2, 1e-3), "ez")
    sim.add_probe((ext / 2, ext / 2, 2e-3), "ez")
    return sim


def test_mesh_intelligence_report_compares_uniform_fine_cells(capsys):
    sim = _patch_like_nonuniform_sim()

    report = sim.mesh_intelligence_report()

    captured = capsys.readouterr()
    assert captured.out == ""
    assert isinstance(report, MeshIntelligenceReport)
    assert report.uses_nonuniform is True
    assert report.cells == report.grid_shape[0] * report.grid_shape[1] * report.grid_shape[2]
    assert report.uniform_fine_cells > report.cells
    assert report.cell_savings_factor > 1.0
    assert report.min_cell_size == np.min(sim._dz_profile)


def test_mesh_intelligence_report_uses_segmented_ad_estimate():
    sim = _patch_like_nonuniform_sim()

    report = sim.mesh_intelligence_report(n_steps=10_000, checkpoint_every=1000)

    assert report.ad_memory is not None
    assert report.ad_memory.checkpoint_every == 1000
    assert report.ad_memory.ad_segmented_gb is not None
    assert "segmented AD estimate" in report.recommendation
    assert "legacy step-checkpoint" in report.recommendation


def test_mesh_intelligence_report_serializes_artifact():
    sim = _patch_like_nonuniform_sim()

    report = sim.mesh_intelligence_report(n_steps=10_000, checkpoint_every=1000)
    artifact = report.to_dict()
    parsed = json.loads(report.to_json())

    assert artifact["uses_nonuniform"] is True
    assert artifact["grid_shape"] == list(report.grid_shape)
    assert artifact["uniform_fine_shape"] == list(report.uniform_fine_shape)
    assert artifact["cell_savings_factor"] == report.cell_savings_factor
    assert artifact["ad_memory"] is not None
    assert artifact["ad_memory"]["checkpoint_every"] == 1000
    assert artifact["ad_memory"]["ad_segmented_gb"] == report.ad_memory.ad_segmented_gb
    assert parsed == artifact


def test_mesh_intelligence_report_captures_preflight_issues():
    sim = Simulation(freq_max=30e9, domain=(0.02, 0.02, 0.02), dx=2e-3,
                     cpml_layers=4)
    sim.add_material("fr4", eps_r=4.3)
    sim.add(Box((0.005, 0.005, 0.005), (0.015, 0.015, 0.015)),
            material="fr4")
    sim.add_source((0.010, 0.010, 0.003), "ez")

    report = sim.mesh_intelligence_report()

    assert report.preflight_issues
    assert any("cells per λ_eff" in issue for issue in report.preflight_issues)
    assert "resolve" in report.recommendation
