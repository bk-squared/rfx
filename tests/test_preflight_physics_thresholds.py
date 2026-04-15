"""Issue #37: preflight thresholds must be physics-based, not cell-count.

Validated configurations (e.g. 05_patch_antenna) should produce no false
under-resolved warnings. Only genuine under-resolution should warn.
"""

from __future__ import annotations

import numpy as np

from rfx import Simulation, Box


def _issues(sim):
    return sim.preflight()


def _has(issues, substring):
    return any(substring in i for i in issues)


def test_thin_pec_sheet_is_silent():
    """1-cell PEC on dx=0.5mm (half-wavelength-fraction) should not warn."""
    sim = Simulation(freq_max=10e9, domain=(0.01, 0.01, 0.01), dx=0.5e-3,
                     cpml_layers=4)
    sim.add_source((0.005, 0.005, 0.002), "ez")
    sim.add_probe((0.005, 0.005, 0.005), "ez")
    sim.add(Box((0.003, 0.003, 0.005), (0.007, 0.007, 0.0055)), material="pec")
    issues = _issues(sim)
    assert not _has(issues, "PEC volume"), (
        f"1-cell PEC should not trigger a volume under-resolved warning; "
        f"issues: {issues!r}"
    )


def test_partial_pec_volume_warns():
    """3-cell PEC extent is the partial-volume case — should warn."""
    sim = Simulation(freq_max=10e9, domain=(0.02, 0.02, 0.02), dx=1e-3,
                     cpml_layers=4)
    sim.add_source((0.01, 0.01, 0.002), "ez")
    sim.add(Box((0.005, 0.005, 0.005), (0.010, 0.010, 0.008)),
            material="pec")
    issues = _issues(sim)
    assert _has(issues, "PEC volume"), (
        f"3-cell PEC volume should warn; issues: {issues!r}"
    )


def test_fine_dielectric_is_silent():
    """Dielectric with ≥10 cells per λ_eff should not warn."""
    sim = Simulation(freq_max=2.4e9, domain=(0.08, 0.08, 0.04), dx=1e-3,
                     cpml_layers=4)
    sim.add_material("fr4", eps_r=4.3)
    # 60x60x1.5mm substrate at dx=1mm: λ_eff/dx ≈ 60mm/1mm = 60 → silent.
    sim.add(Box((0.010, 0.010, 0.012),
                (0.070, 0.070, 0.0135)), material="fr4")
    sim.add_source((0.04, 0.04, 0.013), "ez")
    issues = _issues(sim)
    assert not _has(issues, "cells per λ_eff"), (
        f"FR4 at 2.4 GHz with dx=1mm should be silent; issues: {issues!r}"
    )


def test_coarse_dielectric_warns():
    """Dielectric with dx near λ_eff should warn."""
    sim = Simulation(freq_max=30e9, domain=(0.02, 0.02, 0.02), dx=2e-3,
                     cpml_layers=4)
    sim.add_material("fr4", eps_r=4.3)
    sim.add(Box((0.005, 0.005, 0.005), (0.015, 0.015, 0.015)),
            material="fr4")
    sim.add_source((0.010, 0.010, 0.003), "ez")
    issues = _issues(sim)
    assert _has(issues, "cells per λ_eff"), (
        f"Coarse FR4 should warn; issues: {issues!r}"
    )
