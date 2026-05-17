"""Public-API regression tests for guarded-envelope subgrid validation.

These tests exercise the public ``rfx`` surface only — ``Simulation``,
``add_refinement``, and ``validate_subgrid`` — never ``scripts/``. They cover
both an accepted guarded one-sided z-slab envelope and rejected out-of-envelope
configurations, asserting the fail-closed validation lane returns a structured
``SubgridValidationReport`` with a non-empty ``SubgridValidationIssue`` list.

No test calls ``run()`` on a refined simulation: this branch carries the
guarded subgrid core but not the SBP-SAT runner.
"""

from __future__ import annotations

import pytest

from rfx import (
    Box,
    GaussianPulse,
    Simulation,
    SubgridValidationIssue,
    SubgridValidationReport,
)


def _guarded_envelope_simulation() -> Simulation:
    """Return an accepted guarded one-sided z-slab vacuum envelope config.

    The refined slab touches the z-lo PEC face, x/y stay closed PEC with no
    CPML, and the source/probe sit well inside the fine grid.
    """
    sim = Simulation(
        freq_max=6e9,
        domain=(0.04, 0.04, 0.02),
        boundary="pec",
        cpml_layers=0,
        dx=0.002,
    )
    sim.add_refinement((0.0, 0.012), ratio=4)
    sim.add_source(
        (0.02, 0.02, 0.002),
        "ez",
        waveform=GaussianPulse(f0=3e9, bandwidth=0.5),
    )
    sim.add_probe((0.02, 0.02, 0.003), "ez")
    return sim


def test_guarded_envelope_config_is_supported():
    """An accepted guarded one-sided z-slab config validates as supported."""
    sim = _guarded_envelope_simulation()
    report = sim.validate_subgrid()
    assert isinstance(report, SubgridValidationReport)
    assert report.supported is True
    assert report.errors == ()
    assert report.support_level == "production-z-slab-guarded-boundary-vacuum-envelope"


def test_supported_report_has_serializable_region():
    """A supported report exposes a structured, JSON-serializable artifact."""
    sim = _guarded_envelope_simulation()
    report = sim.validate_subgrid()
    assert report.region is not None
    payload = report.to_dict()
    assert payload["supported"] is True
    assert isinstance(payload["region"], dict)
    # raise_if_unsupported returns self when the config is inside the envelope.
    assert report.raise_if_unsupported() is report


def test_centered_two_interface_slab_is_rejected():
    """A centered/two-interface slab is out-of-envelope and rejected."""
    sim = Simulation(
        freq_max=6e9,
        domain=(0.04, 0.04, 0.02),
        boundary="pec",
        cpml_layers=0,
        dx=0.002,
    )
    # Slab touches neither physical z face -> two artificial interfaces.
    sim.add_refinement((0.006, 0.014), ratio=4)
    sim.add_source(
        (0.02, 0.02, 0.010),
        "ez",
        waveform=GaussianPulse(f0=3e9, bandwidth=0.5),
    )
    sim.add_probe((0.02, 0.02, 0.011), "ez")

    report = sim.validate_subgrid()
    assert isinstance(report, SubgridValidationReport)
    assert report.supported is False
    assert len(report.errors) >= 1
    assert all(isinstance(issue, SubgridValidationIssue) for issue in report.issues)
    codes = {issue.code for issue in report.errors}
    assert "z_slab_requires_guarded_boundary" in codes
    with pytest.raises(ValueError):
        report.raise_if_unsupported()


def test_material_jump_at_artificial_interface_is_rejected():
    """A material discontinuity near the artificial z interface fails closed."""
    sim = Simulation(
        freq_max=6e9,
        domain=(0.04, 0.04, 0.02),
        boundary="pec",
        cpml_layers=0,
        dx=0.002,
    )
    sim.add_refinement((0.0, 0.010), ratio=4)
    sim.add_material("diel", eps_r=4.0)
    # Dielectric fills the coarse region just above the fine z-hi interface,
    # creating an eps_r jump at an artificial coarse/fine interface.
    sim.add(Box((0.0, 0.0, 0.010), (0.04, 0.04, 0.020)), material="diel")
    sim.add_source(
        (0.02, 0.02, 0.002),
        "ez",
        waveform=GaussianPulse(f0=3e9, bandwidth=0.5),
    )
    sim.add_probe((0.02, 0.02, 0.003), "ez")

    report = sim.validate_subgrid()
    assert isinstance(report, SubgridValidationReport)
    assert report.supported is False
    assert len(report.errors) >= 1
    codes = {issue.code for issue in report.errors}
    assert codes & {
        "material_jump_at_zhi_interface",
        "material_transition_near_artificial_interface",
    }


def test_research_mode_does_not_fault_on_centered_slab():
    """validation='research' surfaces issues without blocking the lane."""
    sim = Simulation(
        freq_max=6e9,
        domain=(0.04, 0.04, 0.02),
        boundary="pec",
        cpml_layers=0,
        dx=0.002,
    )
    sim.add_refinement((0.006, 0.014), ratio=4, validation="research")
    sim.add_source(
        (0.02, 0.02, 0.010),
        "ez",
        waveform=GaussianPulse(f0=3e9, bandwidth=0.5),
    )
    sim.add_probe((0.02, 0.02, 0.011), "ez")

    report = sim.validate_subgrid()
    assert isinstance(report, SubgridValidationReport)
    assert report.mode == "research"
    # Research mode keeps the same report structure and stays serializable.
    assert isinstance(report.to_json(), str)
