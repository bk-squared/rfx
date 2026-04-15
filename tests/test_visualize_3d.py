"""Issue #38: 3D structure + far-field visualisation API."""

from __future__ import annotations

import numpy as np
import pytest

from rfx import Simulation, Box
from rfx.sources.sources import GaussianPulse


pytest.importorskip("plotly")

from rfx.visualize import (  # noqa: E402
    visualize_structure,
    visualize_farfield_3d,
)


def _patch_sim():
    sim = Simulation(freq_max=4e9, domain=(0.08, 0.075, 0.04),
                     dx=1e-3, boundary="cpml", cpml_layers=4)
    sim.add_material("fr4", eps_r=4.3)
    sim.add(Box((0.010, 0.010, 0.011), (0.070, 0.065, 0.012)), material="pec")
    sim.add(Box((0.010, 0.010, 0.012), (0.070, 0.065, 0.0135)), material="fr4")
    sim.add(Box((0.025, 0.018, 0.0135), (0.054, 0.057, 0.0145)), material="pec")
    sim.add_source((0.030, 0.0375, 0.013), "ez",
                   waveform=GaussianPulse(f0=2.4e9, bandwidth=0.8))
    return sim


def test_visualize_structure_returns_plotly_figure():
    sim = _patch_sim()
    fig = visualize_structure(sim)
    # Must have one trace per geometry box + source scatter + CPML wireframe.
    # Geometry = 3 (pec/fr4/pec), source scatter = 1, CPML = 1 → at least 5.
    assert len(fig.data) >= 5
    names = {tr.name for tr in fig.data}
    assert "pec" in names and "fr4" in names
    assert any("domain" in n or "CPML" in n for n in names)


def test_visualize_structure_respects_toggles():
    sim = _patch_sim()
    fig = visualize_structure(
        sim, include_cpml=False, include_sources=False, include_ntff=False,
    )
    names = {tr.name for tr in fig.data}
    assert not any("domain" in n or "CPML" in n for n in names)
    assert not any("source" in (n or "").lower() for n in names)


def test_visualize_farfield_requires_ntff_data():
    sim = _patch_sim()

    class _R:  # minimal result stub
        ntff_data = None
        ntff_box = None
        grid = None

    with pytest.raises(ValueError, match="ntff_data"):
        visualize_farfield_3d(_R(), sim=sim)
