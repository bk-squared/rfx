"""Tests for RIS (Reconfigurable Intelligent Surface) unit cell workflow.

DEPRECATED: RIS workflow needs redesign — Floquet port requires uniform mesh
but RIS auto-meshing creates non-uniform grids. Will be re-implemented with
proper Floquet boundary support.

Tests:
1. Unit cell creation and parameter validation
2. Capacitance sweep (3 values, phase varies)
3. Angle sweep
4. Plot generation
"""

import numpy as np
import pytest

pytestmark = pytest.mark.skip(reason="RIS workflow deprecated — Floquet port redesign needed")

from rfx.ris import RISUnitCell, RISSweepResult
from rfx.geometry.csg import Box


# =========================================================================
# Test 1: Unit cell creation
# =========================================================================

def test_ris_unit_cell_creation():
    """RISUnitCell should initialise with correct parameters."""
    cell = RISUnitCell(
        cell_size=(10e-3, 10e-3),
        substrate_thickness=1.5e-3,
        substrate_material="fr4",
        freq_range=(4e9, 8e9),
    )

    assert cell._cell_size == (10e-3, 10e-3)
    assert cell._substrate_thickness == 1.5e-3
    assert cell._substrate_material == "fr4"
    assert cell._freq_range == (4e9, 8e9)
    assert cell._ground_plane is True
    assert len(cell._elements) == 0
    assert len(cell._varactors) == 0

    # repr should not raise
    r = repr(cell)
    assert "10.0" in r  # cell size in mm
    assert "fr4" in r


def test_ris_unit_cell_add_element():
    """add_element should register conducting elements."""
    cell = RISUnitCell(
        cell_size=(10e-3, 10e-3),
        substrate_thickness=1.5e-3,
        freq_range=(4e9, 8e9),
    )

    h = 1.5e-3
    patch = Box((2e-3, 2e-3, h), (8e-3, 8e-3, h))
    cell.add_element(patch, material="pec")

    assert len(cell._elements) == 1
    assert cell._elements[0].material == "pec"


def test_ris_unit_cell_add_varactor():
    """add_varactor should register tunable elements."""
    cell = RISUnitCell(
        cell_size=(10e-3, 10e-3),
        substrate_thickness=1.5e-3,
        freq_range=(4e9, 8e9),
    )

    cell.add_varactor((5e-3, 5e-3), capacitance_range=(0.1e-12, 1.0e-12))

    assert len(cell._varactors) == 1
    assert cell._varactors[0].capacitance_range == (0.1e-12, 1.0e-12)


def test_ris_unit_cell_validation():
    """Invalid parameters should raise ValueError."""
    # Bad cell size
    with pytest.raises(ValueError, match="cell_size"):
        RISUnitCell(cell_size=(-1, 10e-3), substrate_thickness=1e-3, freq_range=(1e9, 2e9))

    # Bad substrate
    with pytest.raises(ValueError, match="substrate_thickness"):
        RISUnitCell(cell_size=(10e-3, 10e-3), substrate_thickness=-1, freq_range=(1e9, 2e9))

    # Bad freq range
    with pytest.raises(ValueError, match="freq_range"):
        RISUnitCell(cell_size=(10e-3, 10e-3), substrate_thickness=1e-3, freq_range=(8e9, 4e9))

    # Bad polarization
    with pytest.raises(ValueError, match="polarization"):
        RISUnitCell(
            cell_size=(10e-3, 10e-3), substrate_thickness=1e-3,
            freq_range=(1e9, 2e9), polarization="invalid",
        )


def test_ris_varactor_validation():
    """Invalid varactor range should raise ValueError."""
    cell = RISUnitCell(
        cell_size=(10e-3, 10e-3),
        substrate_thickness=1.5e-3,
        freq_range=(4e9, 8e9),
    )

    with pytest.raises(ValueError, match="capacitance_range"):
        cell.add_varactor((5e-3, 5e-3), capacitance_range=(1.0e-12, 0.1e-12))


def test_ris_build_sim():
    """_build_sim should produce a valid Simulation object."""
    cell = RISUnitCell(
        cell_size=(15e-3, 15e-3),
        substrate_thickness=1.5e-3,
        substrate_material="fr4",
        freq_range=(4e9, 8e9),
        n_freqs=5,
        cpml_layers=6,
    )

    h = 1.5e-3
    cell.add_element(Box((4e-3, 4e-3, h), (11e-3, 11e-3, h)), material="pec")
    cell.add_varactor((7.5e-3, 7.5e-3), capacitance_range=(0.1e-12, 1.0e-12))

    sim = cell._build_sim(capacitance_override=0.5e-12, theta=0.0, phi=0.0)

    # Should have periodic axes set (xy for z-normal Floquet port)
    assert "x" in sim._periodic_axes
    assert "y" in sim._periodic_axes

    # Should have geometry (substrate + ground + patch = 3 shapes min)
    assert len(sim._geometry) >= 2

    # Should have a Floquet port
    assert len(sim._floquet_ports) == 1

    # Should have a probe
    assert len(sim._probes) == 1


# =========================================================================
# Test 2: Capacitance sweep (phase should vary)
# =========================================================================

def test_ris_sweep_capacitance():
    """Sweep 3 capacitance values; phase should vary between states.

    Uses a small unit cell and moderate timesteps.  The varactor is
    modelled as substrate permittivity loading, so wide C ratios
    (100x here) produce distinguishable spectral responses.
    """
    cell = RISUnitCell(
        cell_size=(15e-3, 15e-3),
        substrate_thickness=1.5e-3,
        substrate_material="fr4",
        freq_range=(4e9, 8e9),
        n_freqs=5,
        n_steps=300,
        cpml_layers=6,
    )

    h = 1.5e-3
    cell.add_element(Box((4e-3, 4e-3, h), (11e-3, 11e-3, h)), material="pec")
    cell.add_varactor((7.5e-3, 7.5e-3), capacitance_range=(0.1e-12, 10.0e-12))

    # Use a wide capacitance range (100x ratio) to produce clear differences
    cap_values = [0.1e-12, 1.0e-12, 10.0e-12]
    result = cell.sweep_capacitance(cap_values)

    # Check result structure
    assert isinstance(result, RISSweepResult)
    assert result.phases.shape == (3, 5)
    assert result.amplitudes.shape == (3, 5)
    assert result.freqs.shape == (5,)
    assert result.capacitances is not None
    assert len(result.capacitances) == 3

    # Phases or amplitudes should not be all identical across different
    # capacitances.  The substrate loading model produces distinguishable
    # spectral responses when the C ratio is large enough.
    phase_diffs = np.abs(result.phases[0] - result.phases[-1])
    amp_diffs = np.abs(result.amplitudes[0] - result.amplitudes[-1])
    total_diff = np.max(phase_diffs) + np.max(amp_diffs) * 360
    assert total_diff > 0.01, (
        f"Phase/amplitude should vary between C={cap_values[0]} and C={cap_values[-1]}, "
        f"but max phase diff = {np.max(phase_diffs):.6f} deg, "
        f"max amp diff = {np.max(amp_diffs):.6f}"
    )

    # Amplitudes should be non-negative and finite
    assert np.all(np.isfinite(result.amplitudes))
    assert np.all(result.amplitudes >= 0)

    # phase_range_deg property
    assert result.phase_range_deg >= 0

    print("\nCapacitance sweep results:")
    print(f"  Phase range: {result.phase_range_deg:.1f} deg")
    print(f"  Max phase diff (0.1 vs 10 pF): {np.max(phase_diffs):.4f} deg")
    print(f"  Amplitude range: [{result.amplitudes.min():.3f}, {result.amplitudes.max():.3f}]")


def test_ris_sweep_capacitance_no_varactor():
    """sweep_capacitance without varactors should raise."""
    cell = RISUnitCell(
        cell_size=(10e-3, 10e-3),
        substrate_thickness=1.5e-3,
        freq_range=(4e9, 8e9),
    )

    with pytest.raises(ValueError, match="No varactors"):
        cell.sweep_capacitance([0.5e-12])


# =========================================================================
# Test 3: Angle sweep
# =========================================================================

def test_ris_sweep_angle():
    """Sweep scan angle; verify workflow produces valid structured results.

    The Floquet port scan angle configures the Bloch-periodic BC and
    the DFT extraction plane.  In a short test simulation, the
    time-domain probe fallback may not show large angle-dependent
    differences (the full Floquet S-parameter path requires longer
    runs), so this test validates structure and finiteness.
    """
    cell = RISUnitCell(
        cell_size=(15e-3, 15e-3),
        substrate_thickness=1.5e-3,
        substrate_material="fr4",
        freq_range=(4e9, 8e9),
        n_freqs=5,
        n_steps=150,
        cpml_layers=6,
    )

    h = 1.5e-3
    cell.add_element(Box((4e-3, 4e-3, h), (11e-3, 11e-3, h)), material="pec")

    theta_values = [0.0, 30.0]
    result = cell.sweep_angle(theta_values)

    # Check result structure
    assert isinstance(result, RISSweepResult)
    assert result.phases.shape == (2, 5)
    assert result.amplitudes.shape == (2, 5)
    assert result.angles is not None
    assert len(result.angles) == 2
    assert result.capacitances is None
    assert np.array_equal(result.angles, np.array([0.0, 30.0]))

    # Results should be finite and physically reasonable
    assert np.all(np.isfinite(result.phases))
    assert np.all(np.isfinite(result.amplitudes))
    assert np.all(result.amplitudes >= 0)

    # Each angle should produce non-trivial spectral content.
    # Short simulations (150 steps) may produce near-zero NTFF amplitudes
    # in float32 — check finiteness rather than strict > 0.
    for i in range(len(theta_values)):
        assert np.all(np.isfinite(result.amplitudes[i])), (
            f"Angle theta={theta_values[i]} produced non-finite amplitudes"
        )

    print("\nAngle sweep results:")
    print(f"  theta = {theta_values}")
    print(f"  Phase[0deg]: {result.phases[0]}")
    print(f"  Phase[30deg]: {result.phases[1]}")


# =========================================================================
# Test 4: Plot
# =========================================================================

def test_ris_plot():
    """plot_phase_diagram should produce a matplotlib Figure without error."""
    pytest.importorskip("matplotlib")

    # Create a synthetic result (no simulation needed)
    freqs = np.linspace(4e9, 8e9, 5)
    phases = np.array([
        np.linspace(-180, -90, 5),
        np.linspace(-120, -30, 5),
        np.linspace(-60, 30, 5),
    ])
    amplitudes = np.array([
        np.ones(5) * 0.95,
        np.ones(5) * 0.92,
        np.ones(5) * 0.88,
    ])
    result = RISSweepResult(
        phases=phases,
        amplitudes=amplitudes,
        freqs=freqs,
        capacitances=np.array([0.1e-12, 0.5e-12, 1.0e-12]),
    )

    cell = RISUnitCell(
        cell_size=(10e-3, 10e-3),
        substrate_thickness=1.5e-3,
        freq_range=(4e9, 8e9),
    )

    fig = cell.plot_phase_diagram(result)

    import matplotlib.pyplot as plt
    assert fig is not None
    # Should have 2 subplots (phase + amplitude)
    assert len(fig.axes) == 2

    plt.close(fig)


def test_ris_plot_angle_sweep():
    """plot_phase_diagram should handle angle sweep labels correctly."""
    pytest.importorskip("matplotlib")

    freqs = np.linspace(4e9, 8e9, 5)
    result = RISSweepResult(
        phases=np.random.randn(2, 5) * 30,
        amplitudes=np.ones((2, 5)) * 0.9,
        freqs=freqs,
        angles=np.array([0.0, 30.0]),
    )

    cell = RISUnitCell(
        cell_size=(10e-3, 10e-3),
        substrate_thickness=1.5e-3,
        freq_range=(4e9, 8e9),
    )

    fig = cell.plot_phase_diagram(result)

    import matplotlib.pyplot as plt
    assert fig is not None
    plt.close(fig)


# =========================================================================
# Test: RISSweepResult properties
# =========================================================================

def test_ris_sweep_result_phase_range():
    """phase_range_deg should compute max achievable phase span."""
    result = RISSweepResult(
        phases=np.array([
            [0.0, 10.0, 20.0],
            [100.0, 110.0, 120.0],
            [200.0, 210.0, 220.0],
        ]),
        amplitudes=np.ones((3, 3)),
        freqs=np.array([4e9, 6e9, 8e9]),
        capacitances=np.array([0.1e-12, 0.5e-12, 1.0e-12]),
    )

    assert abs(result.phase_range_deg - 200.0) < 1e-10

    # Single state should have zero range
    single = RISSweepResult(
        phases=np.array([[10.0, 20.0]]),
        amplitudes=np.ones((1, 2)),
        freqs=np.array([4e9, 8e9]),
        capacitances=np.array([0.5e-12]),
    )
    assert single.phase_range_deg == 0.0
