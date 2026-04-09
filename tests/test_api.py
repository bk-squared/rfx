"""Tests for the agent-friendly high-level API (Stage 4).

Validates that:
1. Simulation builder creates valid grids and materials
2. Named materials and library materials work
3. Geometry rasterization through the builder works
4. Port + probe simulation produces non-zero results
5. DFT plane probes work through the API
6. TFSF plane-wave setup works through the API
7. S-parameter extraction through the API works
8. Differentiable mode (checkpoint) works through the API
9. Validation catches bad inputs
"""

import numpy as np
import jax
import jax.numpy as jnp
import pytest

import rfx
from rfx.api import Simulation, Result, MATERIAL_LIBRARY
from rfx.geometry.csg import Box
from rfx.sources.sources import GaussianPulse, LumpedPort, setup_lumped_port
from rfx.sources.waveguide_port import (
    extract_waveguide_s21,
    extract_waveguide_sparams,
    waveguide_plane_positions,
)
from rfx.materials.lorentz import lorentz_pole
from rfx.simulation import make_port_source, make_probe, run as low_level_run


def test_basic_simulation():
    """Minimal simulation: PEC cavity with a source and probe."""
    sim = Simulation(freq_max=5e9, domain=(0.03, 0.03, 0.03), boundary="pec")
    sim.add_port((0.015, 0.015, 0.015), "ez")
    sim.add_probe((0.02, 0.02, 0.02), "ez")

    result = sim.run(n_steps=50, compute_s_params=False)

    assert isinstance(result, Result)
    assert result.time_series.shape == (50, 1)
    peak = float(jnp.max(jnp.abs(result.time_series)))
    assert peak > 0, "Probe should detect non-zero field"


def test_upml_boundary_runs_through_api():
    """UPML boundary should be accepted for the uniform-grid Yee path."""
    sim = Simulation(
        freq_max=6e9,
        domain=(0.04, 0.04, 0.004),
        boundary="upml",
        cpml_layers=6,
        dx=0.002,
        mode="2d_tmz",
    )
    sim.add_source((0.01, 0.02, 0.0), "ez", waveform=GaussianPulse(f0=3e9, bandwidth=0.5))
    sim.add_probe((0.026, 0.02, 0.0), "ez")

    result = sim.run(n_steps=80, compute_s_params=False)

    assert result.time_series.shape == (80, 1)
    assert float(jnp.max(jnp.abs(result.time_series[:, 0]))) > 0.0


def test_upml_rejects_subgridding_refinement():
    """UPML v1 should fail loudly on subgridding/refinement."""
    sim = Simulation(
        freq_max=6e9,
        domain=(0.04, 0.04, 0.02),
        boundary="upml",
        cpml_layers=6,
        dx=0.002,
    )
    sim.add_refinement((0.004, 0.008), ratio=2)
    sim.add_source((0.01, 0.02, 0.01), "ez", waveform=GaussianPulse(f0=3e9, bandwidth=0.5))
    sim.add_probe((0.026, 0.02, 0.01), "ez")

    with pytest.raises(ValueError, match="subgridding/refinement"):
        sim.run(n_steps=40, compute_s_params=False)


def test_upml_rejects_distributed_execution():
    """UPML v1 should fail loudly instead of leaking into distributed mode."""
    devices = jax.devices()
    if len(devices) < 2:
        pytest.skip("requires >=2 JAX devices")

    sim = Simulation(
        freq_max=6e9,
        domain=(0.04, 0.04, 0.004),
        boundary="upml",
        cpml_layers=6,
        dx=0.002,
        mode="2d_tmz",
    )
    sim.add_source((0.01, 0.02, 0.0), "ez", waveform=GaussianPulse(f0=3e9, bandwidth=0.5))
    sim.add_probe((0.026, 0.02, 0.0), "ez")

    with pytest.raises(ValueError, match="distributed execution"):
        sim.run(n_steps=40, compute_s_params=False, devices=devices[:2])


def test_named_material():
    """Custom named material is applied to geometry."""
    sim = Simulation(freq_max=5e9, domain=(0.03, 0.03, 0.03), boundary="pec")
    sim.add_material("dielectric", eps_r=4.0)
    sim.add(Box((0.005, 0.005, 0.005), (0.025, 0.025, 0.025)),
            material="dielectric")
    sim.add_port((0.015, 0.015, 0.015), "ez")
    sim.add_probe((0.02, 0.02, 0.02), "ez")

    result = sim.run(n_steps=50, compute_s_params=False)
    assert float(jnp.max(jnp.abs(result.time_series))) > 0


def test_library_material():
    """Library material (fr4) works without explicit registration."""
    sim = Simulation(freq_max=5e9, domain=(0.03, 0.03, 0.03), boundary="pec")
    sim.add(Box((0, 0, 0), (0.03, 0.03, 0.001)), material="fr4")
    sim.add_probe((0.015, 0.015, 0.015), "ez")

    result = sim.run(n_steps=30, compute_s_params=False)
    assert result.time_series.shape[0] == 30


def test_tfsf_source_through_api():
    """High-level API should drive the compiled TFSF path."""
    sim = Simulation(
        freq_max=8e9,
        domain=(0.08, 0.006, 0.006),
        boundary="cpml",
        cpml_layers=8,
        dx=0.001,
    )
    sim.add_tfsf_source(f0=4e9, bandwidth=0.5, amplitude=1.0, margin=3)
    sim.add_probe((0.02, 0.003, 0.003), "ez")

    result = sim.run(n_steps=120, compute_s_params=False)

    assert result.time_series.shape == (120, 1)
    peak = float(jnp.max(jnp.abs(result.time_series[:, 0])))
    assert peak > 0.0, "Probe should detect the injected plane wave"


def test_tfsf_ey_polarization_through_api():
    """High-level API should support Ey-polarized normal-incidence TFSF."""
    sim = Simulation(
        freq_max=8e9,
        domain=(0.08, 0.006, 0.006),
        boundary="cpml",
        cpml_layers=8,
        dx=0.001,
    )
    sim.add_tfsf_source(f0=4e9, bandwidth=0.5, amplitude=1.0, margin=3, polarization="ey")
    sim.add_probe((0.02, 0.003, 0.003), "ey")

    result = sim.run(n_steps=120, compute_s_params=False)

    assert result.time_series.shape == (120, 1)
    peak = float(jnp.max(jnp.abs(result.time_series[:, 0])))
    assert peak > 0.0, "Probe should detect the Ey-polarized plane wave"


def test_tfsf_negative_x_direction_through_api():
    """High-level API should support reverse-propagating normal-incidence TFSF."""
    sim = Simulation(
        freq_max=8e9,
        domain=(0.08, 0.006, 0.006),
        boundary="cpml",
        cpml_layers=8,
        dx=0.001,
    )
    sim.add_tfsf_source(
        f0=4e9, bandwidth=0.5, amplitude=1.0, margin=3,
        polarization="ez", direction="-x",
    )
    sim.add_probe((0.02, 0.003, 0.003), "ez")

    result = sim.run(n_steps=120, compute_s_params=False)

    assert result.time_series.shape == (120, 1)
    peak = float(jnp.max(jnp.abs(result.time_series[:, 0])))
    assert peak > 0.0, "Probe should detect the reverse-propagating plane wave"


def test_tfsf_oblique_incidence_through_api():
    """High-level API should support oblique Ez-polarized TFSF."""
    sim = Simulation(
        freq_max=5e9,
        domain=(0.12, 0.12, 0.006),
        boundary="cpml",
        cpml_layers=8,
        dx=0.004,
    )
    sim.add_tfsf_source(
        f0=5e9, bandwidth=0.2, amplitude=1.0,
        margin=3, polarization="ez", angle_deg=30.0,
    )
    sim.add_probe((0.04, 0.06, 0.003), "ez")

    result = sim.run(n_steps=120, compute_s_params=False)

    assert result.time_series.shape == (120, 1)
    peak = float(jnp.max(jnp.abs(result.time_series[:, 0])))
    assert peak > 0.0, "Probe should detect the obliquely incident plane wave"


def test_dft_plane_probe_through_api():
    """High-level API should expose plane DFT monitors."""
    sim = Simulation(freq_max=5e9, domain=(0.03, 0.03, 0.03), boundary="pec")
    sim.add_port((0.015, 0.015, 0.015), "ez")
    sim.add_dft_plane_probe(
        axis="x",
        coordinate=0.015,
        component="ez",
        freqs=jnp.array([2e9, 3e9, 4e9]),
        name="mid_ez",
    )

    result = sim.run(n_steps=40, compute_s_params=False)

    assert result.dft_planes is not None
    assert "mid_ez" in result.dft_planes
    plane = result.dft_planes["mid_ez"]
    assert plane.accumulator.shape[0] == 3
    assert plane.accumulator.shape[1:] == result.state.ez.shape[1:]
    assert float(jnp.max(jnp.abs(plane.accumulator))) > 0.0


def test_waveguide_port_through_api():
    """High-level API should expose the rectangular-waveguide workflow."""
    sim = Simulation(
        freq_max=10e9,
        domain=(0.12, 0.04, 0.02),
        boundary="cpml",
        cpml_layers=10,
        dx=0.002,
    )
    sim.add_waveguide_port(
        0.01,
        mode=(1, 0),
        mode_type="TE",
        freqs=jnp.linspace(4.5e9, 8e9, 12),
        f0=6e9,
        probe_offset=15,
        ref_offset=3,
        name="wg0",
    )

    result = sim.run(n_steps=300, compute_s_params=False)

    assert result.waveguide_ports is not None
    assert "wg0" in result.waveguide_ports
    s21 = np.abs(np.array(extract_waveguide_s21(result.waveguide_ports["wg0"])))
    assert np.max(s21) > 0.05, "Waveguide response should be nontrivial above cutoff"


def test_waveguide_port_rejects_unsupported_configuration():
    """Waveguide API should fail loudly on unsupported setups."""
    sim_pec = Simulation(freq_max=10e9, domain=(0.12, 0.04, 0.02), boundary="pec")
    with pytest.raises(ValueError, match="boundary='cpml'"):
        sim_pec.add_waveguide_port(0.01)

    sim_tfsf = Simulation(
        freq_max=10e9,
        domain=(0.12, 0.04, 0.02),
        boundary="cpml",
        cpml_layers=10,
        dx=0.002,
    )
    sim_tfsf.add_tfsf_source()
    with pytest.raises(ValueError, match="TFSF"):
        sim_tfsf.add_waveguide_port(0.01)

    sim_geom = Simulation(
        freq_max=10e9,
        domain=(0.12, 0.04, 0.02),
        boundary="cpml",
        cpml_layers=10,
        dx=0.002,
    )
    sim_geom.add(Box((0, 0, 0), (0.01, 0.01, 0.01)), material="vacuum")
    sim_geom.add_waveguide_port(0.01)

    sim_multi = Simulation(
        freq_max=10e9,
        domain=(0.12, 0.04, 0.02),
        boundary="cpml",
        cpml_layers=10,
        dx=0.002,
    )
    sim_multi.add_waveguide_port(0.01, direction="+x")
    sim_multi.add_waveguide_port(0.09, direction="-x")
    with pytest.raises(ValueError, match="Simulation.run\\(\\) supports only a single waveguide port"):
        sim_multi.run(n_steps=40, compute_s_params=False)


def test_waveguide_port_geometry_changes_response():
    """Waveguide API should support non-empty guide geometries."""
    base_kwargs = dict(
        freq_max=10e9,
        domain=(0.12, 0.04, 0.02),
        boundary="cpml",
        cpml_layers=10,
        dx=0.002,
    )

    sim_empty = Simulation(**base_kwargs)
    sim_empty.add_waveguide_port(
        0.01,
        mode=(1, 0),
        mode_type="TE",
        freqs=jnp.linspace(4.5e9, 8e9, 12),
        f0=6e9,
        probe_offset=15,
        ref_offset=3,
        name="wg0",
    )

    sim_loaded = Simulation(**base_kwargs)
    sim_loaded.add_material("diel", eps_r=2.5)
    sim_loaded.add(Box((0.04, 0.0, 0.0), (0.08, 0.04, 0.02)), material="diel")
    sim_loaded.add_waveguide_port(
        0.01,
        mode=(1, 0),
        mode_type="TE",
        freqs=jnp.linspace(4.5e9, 8e9, 12),
        f0=6e9,
        probe_offset=15,
        ref_offset=3,
        name="wg0",
    )

    empty = sim_empty.run(n_steps=300, compute_s_params=False)
    loaded = sim_loaded.run(n_steps=300, compute_s_params=False)

    s21_empty = np.array(extract_waveguide_s21(empty.waveguide_ports["wg0"]))
    s21_loaded = np.array(extract_waveguide_s21(loaded.waveguide_ports["wg0"]))
    assert np.max(np.abs(s21_empty - s21_loaded)) > 1e-3


def test_waveguide_two_port_s_matrix_through_api():
    """High-level API should assemble a true two-port waveguide S-matrix."""
    sim = Simulation(
        freq_max=10e9,
        domain=(0.12, 0.04, 0.02),
        boundary="cpml",
        cpml_layers=10,
        dx=0.002,
    )
    sim.add_waveguide_port(
        0.01,
        direction="+x",
        mode=(1, 0),
        mode_type="TE",
        freqs=jnp.linspace(4.5e9, 8e9, 12),
        f0=6e9,
        ref_offset=3,
        probe_offset=15,
        name="left",
    )
    sim.add_waveguide_port(
        0.09,
        direction="-x",
        mode=(1, 0),
        mode_type="TE",
        freqs=jnp.linspace(4.5e9, 8e9, 12),
        f0=6e9,
        ref_offset=3,
        probe_offset=15,
        name="right",
    )

    result = sim.compute_waveguide_s_matrix(num_periods=30)

    assert result.s_params.shape == (2, 2, 12)
    assert result.port_names == ("left", "right")
    assert result.port_directions == ("+x", "-x")
    s11 = result.s_params[0, 0, :]
    s21 = result.s_params[1, 0, :]
    s22 = result.s_params[1, 1, :]
    s12 = result.s_params[0, 1, :]
    recip_err = np.mean(
        np.abs(np.abs(s21) - np.abs(s12))
        / np.maximum(0.5 * (np.abs(s21) + np.abs(s12)), 1e-8)
    )
    assert np.mean(np.abs(s21)) > 0.5
    assert np.mean(np.abs(s12)) > 0.5
    assert np.mean(np.abs(s11)) < 1.0
    assert np.mean(np.abs(s22)) < 1.0
    assert recip_err < 0.2


def test_waveguide_multiport_same_direction_requires_shared_boundary_plane():
    """Ports on one boundary direction must share a single boundary plane."""
    sim = Simulation(
        freq_max=10e9,
        domain=(0.12, 0.04, 0.02),
        boundary="cpml",
        cpml_layers=10,
        dx=0.002,
    )
    sim.add_waveguide_port(0.01, direction="+x", name="left")
    sim.add_waveguide_port(0.09, direction="+x", name="right")
    with pytest.raises(ValueError, match="boundary \\+x must share one boundary plane"):
        sim.compute_waveguide_s_matrix(n_steps=80)


def test_waveguide_four_port_parallel_guides_through_api():
    """Disjoint left/right apertures should support a 4-port parallel-guide S-matrix."""
    sim = Simulation(
        freq_max=10e9,
        domain=(0.12, 0.10, 0.02),
        boundary="cpml",
        cpml_layers=10,
        dx=0.002,
    )
    sim.add(Box((0.0, 0.04, 0.0), (0.12, 0.06, 0.02)), material="pec")
    common = dict(
        mode=(1, 0),
        mode_type="TE",
        freqs=jnp.linspace(4.5e9, 8.0e9, 10),
        f0=6e9,
        ref_offset=3,
        probe_offset=15,
    )
    sim.add_waveguide_port(0.01, y_range=(0.00, 0.04), z_range=(0.00, 0.02), direction="+x", name="left_lo", **common)
    sim.add_waveguide_port(0.01, y_range=(0.06, 0.10), z_range=(0.00, 0.02), direction="+x", name="left_hi", **common)
    sim.add_waveguide_port(0.09, y_range=(0.00, 0.04), z_range=(0.00, 0.02), direction="-x", name="right_lo", **common)
    sim.add_waveguide_port(0.09, y_range=(0.06, 0.10), z_range=(0.00, 0.02), direction="-x", name="right_hi", **common)

    result = sim.compute_waveguide_s_matrix(num_periods=30)
    S = result.s_params
    assert S.shape == (4, 4, 10)
    assert result.port_names == ("left_lo", "left_hi", "right_lo", "right_hi")

    lo_tx = np.mean(np.abs(S[2, 0, :]))
    hi_tx = np.mean(np.abs(S[3, 1, :]))
    cross_lo_to_hi = np.mean(np.abs(S[3, 0, :]))
    cross_hi_to_lo = np.mean(np.abs(S[2, 1, :]))

    assert lo_tx > 0.2
    assert hi_tx > 0.2
    assert cross_lo_to_hi < lo_tx
    assert cross_hi_to_lo < hi_tx


def test_waveguide_same_boundary_overlapping_apertures_reject():
    """Ports sharing a boundary plane must have disjoint apertures."""
    sim = Simulation(
        freq_max=10e9,
        domain=(0.12, 0.10, 0.02),
        boundary="cpml",
        cpml_layers=10,
        dx=0.002,
    )
    common = dict(
        mode=(1, 0),
        mode_type="TE",
        freqs=jnp.linspace(4.5e9, 8.0e9, 8),
        f0=6e9,
        ref_offset=3,
        probe_offset=15,
    )
    sim.add_waveguide_port(0.01, y_range=(0.00, 0.05), z_range=(0.00, 0.02), direction="+x", name="a", **common)
    sim.add_waveguide_port(0.01, y_range=(0.04, 0.09), z_range=(0.00, 0.02), direction="+x", name="b", **common)
    sim.add_waveguide_port(0.09, y_range=(0.00, 0.05), z_range=(0.00, 0.02), direction="-x", name="c", **common)
    with pytest.raises(ValueError, match="same \\+x boundary must have disjoint apertures"):
        sim.compute_waveguide_s_matrix(n_steps=120)


def test_waveguide_two_port_y_normal_s_matrix_through_api():
    """Y-normal end ports should produce a reciprocal two-port guide S-matrix."""
    sim = Simulation(
        freq_max=10e9,
        domain=(0.04, 0.12, 0.02),
        boundary="cpml",
        cpml_layers=10,
        dx=0.002,
    )
    common = dict(
        mode=(1, 0),
        mode_type="TE",
        freqs=jnp.linspace(4.5e9, 8e9, 12),
        f0=6e9,
        ref_offset=3,
        probe_offset=15,
    )
    sim.add_waveguide_port(0.01, direction="+y", name="bottom", **common)
    sim.add_waveguide_port(0.09, direction="-y", name="top", **common)

    result = sim.compute_waveguide_s_matrix(num_periods=30)
    s_bottom_top = result.s_params[1, 0, :]
    s_top_bottom = result.s_params[0, 1, :]
    recip_err = np.mean(
        np.abs(np.abs(s_bottom_top) - np.abs(s_top_bottom))
        / np.maximum(0.5 * (np.abs(s_bottom_top) + np.abs(s_top_bottom)), 1e-8)
    )

    assert result.port_directions == ("+y", "-y")
    assert np.mean(np.abs(s_bottom_top)) > 0.5
    assert np.mean(np.abs(s_top_bottom)) > 0.5
    assert recip_err < 0.2


def test_waveguide_branch_junction_mixed_normals_reciprocal_through_api():
    """Mixed x/y boundary ports in a T-junction should remain reciprocal."""
    sim = Simulation(
        freq_max=10e9,
        domain=(0.12, 0.12, 0.02),
        boundary="cpml",
        cpml_layers=10,
        dx=0.002,
    )
    sim.add(Box((0.0, 0.0, 0.0), (0.12, 0.04, 0.02)), material="pec")
    sim.add(Box((0.0, 0.08, 0.0), (0.04, 0.12, 0.02)), material="pec")
    sim.add(Box((0.08, 0.08, 0.0), (0.12, 0.12, 0.02)), material="pec")

    common = dict(
        mode=(1, 0),
        mode_type="TE",
        freqs=jnp.linspace(4.5e9, 8.0e9, 10),
        f0=6e9,
        ref_offset=3,
        probe_offset=15,
        z_range=(0.00, 0.02),
    )
    sim.add_waveguide_port(0.01, y_range=(0.04, 0.08), direction="+x", name="left", **common)
    sim.add_waveguide_port(0.11, y_range=(0.04, 0.08), direction="-x", name="right", **common)
    sim.add_waveguide_port(0.11, x_range=(0.04, 0.08), direction="-y", name="top", **common)

    result = sim.compute_waveguide_s_matrix(num_periods=30)
    S = result.s_params
    assert S.shape == (3, 3, 10)
    assert result.port_directions == ("+x", "-x", "-y")

    left_right = np.mean(np.abs(S[1, 0, :]))
    left_top = np.mean(np.abs(S[2, 0, :]))
    top_left = np.mean(np.abs(S[0, 2, :]))
    top_right = np.mean(np.abs(S[1, 2, :]))

    recip_left_right = np.mean(
        np.abs(np.abs(S[1, 0, :]) - np.abs(S[0, 1, :]))
        / np.maximum(0.5 * (np.abs(S[1, 0, :]) + np.abs(S[0, 1, :])), 1e-8)
    )
    recip_left_top = np.mean(
        np.abs(np.abs(S[2, 0, :]) - np.abs(S[0, 2, :]))
        / np.maximum(0.5 * (np.abs(S[2, 0, :]) + np.abs(S[0, 2, :])), 1e-8)
    )
    recip_right_top = np.mean(
        np.abs(np.abs(S[2, 1, :]) - np.abs(S[1, 2, :]))
        / np.maximum(0.5 * (np.abs(S[2, 1, :]) + np.abs(S[1, 2, :])), 1e-8)
    )

    assert left_right > 0.2
    assert left_top > 0.15
    assert top_left > 0.15
    assert top_right > 0.15
    assert recip_left_right < 0.2
    assert recip_left_top < 0.2
    assert recip_right_top < 0.2


def test_waveguide_sparams_default_output():
    """High-level result should expose calibrated waveguide S-params by default."""
    sim = Simulation(
        freq_max=10e9,
        domain=(0.12, 0.04, 0.02),
        boundary="cpml",
        cpml_layers=10,
        dx=0.002,
    )
    sim.add_waveguide_port(
        0.01,
        mode=(1, 0),
        mode_type="TE",
        freqs=jnp.linspace(4.5e9, 8e9, 12),
        f0=6e9,
        probe_offset=15,
        ref_offset=3,
        name="wg0",
    )

    result = sim.run(n_steps=300, compute_s_params=False)

    assert result.waveguide_ports is not None
    assert result.waveguide_sparams is not None
    raw_cfg = result.waveguide_ports["wg0"]
    calibrated = result.waveguide_sparams["wg0"]
    raw_s11, raw_s21 = extract_waveguide_sparams(raw_cfg)
    np.testing.assert_allclose(calibrated.s11, np.array(raw_s11))
    np.testing.assert_allclose(calibrated.s21, np.array(raw_s21))
    assert calibrated.calibration_preset == "measured"
    assert calibrated.source_plane == pytest.approx(0.01)
    assert calibrated.measured_reference_plane == pytest.approx(0.016)
    assert calibrated.measured_probe_plane == pytest.approx(0.040)
    assert calibrated.reference_plane == pytest.approx(0.016)
    assert calibrated.probe_plane == pytest.approx(0.040)


def test_waveguide_sparams_deembedded_planes():
    """High-level API should de-embed waveguide S-params to requested planes."""
    sim = Simulation(
        freq_max=10e9,
        domain=(0.12, 0.04, 0.02),
        boundary="cpml",
        cpml_layers=10,
        dx=0.002,
    )
    sim.add_waveguide_port(
        0.01,
        mode=(1, 0),
        mode_type="TE",
        freqs=jnp.linspace(4.5e9, 8e9, 12),
        f0=6e9,
        probe_offset=15,
        ref_offset=3,
        reference_plane=0.012,
        probe_plane=0.034,
        name="wg0",
    )

    result = sim.run(n_steps=300, compute_s_params=False)

    raw_cfg = result.waveguide_ports["wg0"]
    calibrated = result.waveguide_sparams["wg0"]
    expected_s11, expected_s21 = extract_waveguide_sparams(
        raw_cfg,
        ref_shift=0.012 - 0.016,
        probe_shift=0.034 - 0.040,
    )
    np.testing.assert_allclose(calibrated.s11, np.array(expected_s11))
    np.testing.assert_allclose(calibrated.s21, np.array(expected_s21))
    assert calibrated.calibration_preset == "explicit"
    assert calibrated.reference_plane == pytest.approx(0.012)
    assert calibrated.probe_plane == pytest.approx(0.034)


def test_waveguide_sparams_source_to_probe_preset():
    """A calibration preset should auto-select practical reporting planes."""
    sim = Simulation(
        freq_max=10e9,
        domain=(0.12, 0.04, 0.02),
        boundary="cpml",
        cpml_layers=10,
        dx=0.002,
    )
    sim.add_waveguide_port(
        0.01,
        mode=(1, 0),
        mode_type="TE",
        freqs=jnp.linspace(4.5e9, 8e9, 12),
        f0=6e9,
        probe_offset=15,
        ref_offset=3,
        calibration_preset="source_to_probe",
        name="wg0",
    )

    result = sim.run(n_steps=300, compute_s_params=False)

    raw_cfg = result.waveguide_ports["wg0"]
    calibrated = result.waveguide_sparams["wg0"]
    expected_s11, expected_s21 = extract_waveguide_sparams(
        raw_cfg,
        ref_shift=0.01 - 0.016,
        probe_shift=0.040 - 0.040,
    )
    assert calibrated.calibration_preset == "source_to_probe"
    assert calibrated.reference_plane == pytest.approx(0.010)
    assert calibrated.probe_plane == pytest.approx(0.040)
    np.testing.assert_allclose(calibrated.s11, np.array(expected_s11))
    np.testing.assert_allclose(calibrated.s21, np.array(expected_s21))


def test_waveguide_sparams_report_snapped_planes_for_non_aligned_input():
    """Result metadata should expose actual snapped planes, not raw requested x values."""
    sim = Simulation(
        freq_max=10e9,
        domain=(0.12, 0.04, 0.02),
        boundary="cpml",
        cpml_layers=10,
        dx=0.002,
    )
    sim.add_waveguide_port(
        0.0109,
        mode=(1, 0),
        mode_type="TE",
        freqs=jnp.linspace(4.5e9, 8e9, 12),
        f0=6e9,
        probe_offset=15,
        ref_offset=3,
        reference_plane=0.0131,
        probe_plane=0.0331,
        name="wg0",
    )

    result = sim.run(n_steps=300, compute_s_params=False)

    raw_cfg = result.waveguide_ports["wg0"]
    calibrated = result.waveguide_sparams["wg0"]
    plane_positions = waveguide_plane_positions(raw_cfg)
    expected_s11, expected_s21 = extract_waveguide_sparams(
        raw_cfg,
        ref_shift=0.0131 - plane_positions["reference"],
        probe_shift=0.0331 - plane_positions["probe"],
    )

    assert plane_positions["source"] == pytest.approx(0.010)
    assert plane_positions["reference"] == pytest.approx(0.016)
    assert plane_positions["probe"] == pytest.approx(0.040)
    assert calibrated.source_plane == pytest.approx(plane_positions["source"])
    assert calibrated.measured_reference_plane == pytest.approx(plane_positions["reference"])
    assert calibrated.measured_probe_plane == pytest.approx(plane_positions["probe"])
    assert calibrated.reference_plane == pytest.approx(0.0131)
    assert calibrated.probe_plane == pytest.approx(0.0331)
    np.testing.assert_allclose(calibrated.s11, np.array(expected_s11))
    np.testing.assert_allclose(calibrated.s21, np.array(expected_s21))


def test_waveguide_port_allows_near_boundary_input_when_snapped_planes_fit():
    """Near-boundary requested positions should validate against snapped planes."""
    sim = Simulation(
        freq_max=10e9,
        domain=(0.12, 0.04, 0.02),
        boundary="cpml",
        cpml_layers=10,
        dx=0.002,
    )
    sim.add_waveguide_port(
        0.0909,
        mode=(1, 0),
        mode_type="TE",
        freqs=jnp.linspace(4.5e9, 8e9, 12),
        f0=6e9,
        probe_offset=15,
        ref_offset=3,
        name="wg0",
    )

    result = sim.run(n_steps=300, compute_s_params=False)
    calibrated = result.waveguide_sparams["wg0"]
    assert calibrated.source_plane == pytest.approx(0.090)
    assert calibrated.measured_reference_plane == pytest.approx(0.096)
    assert calibrated.measured_probe_plane == pytest.approx(0.120)


def test_waveguide_port_rejects_invalid_reporting_plane():
    """Waveguide calibration planes should stay inside the physical x-domain."""
    sim = Simulation(
        freq_max=10e9,
        domain=(0.12, 0.04, 0.02),
        boundary="cpml",
        cpml_layers=10,
        dx=0.002,
    )
    with pytest.raises(ValueError, match="reference_plane .* outside the x-domain"):
        sim.add_waveguide_port(0.01, reference_plane=-0.001)
    with pytest.raises(ValueError, match="probe_plane .* outside the x-domain"):
        sim.add_waveguide_port(0.01, probe_plane=0.121)
    with pytest.raises(ValueError, match="probe_plane must be >?= reference_plane"):
        sim.add_waveguide_port(0.01, reference_plane=0.03, probe_plane=0.02)
    with pytest.raises(ValueError, match="calibration_preset must be one of"):
        sim.add_waveguide_port(0.01, calibration_preset="bad")
    with pytest.raises(ValueError, match="cannot be combined with explicit"):
        sim.add_waveguide_port(0.01, calibration_preset="source_to_probe", reference_plane=0.02)


def test_waveguide_port_rejects_nonfinite_numeric_inputs():
    """High-level waveguide API should reject NaN/inf scalar inputs."""
    sim = Simulation(
        freq_max=10e9,
        domain=(0.12, 0.04, 0.02),
        boundary="cpml",
        cpml_layers=10,
        dx=0.002,
    )
    with pytest.raises(ValueError, match="x_position must be finite"):
        sim.add_waveguide_port(float("nan"))
    with pytest.raises(ValueError, match="bandwidth must be finite"):
        sim.add_waveguide_port(0.01, bandwidth=float("inf"))
    with pytest.raises(ValueError, match="amplitude must be finite"):
        sim.add_waveguide_port(0.01, amplitude=float("nan"))
    with pytest.raises(ValueError, match="f0 must be finite"):
        sim.add_waveguide_port(0.01, f0=float("inf"))
    with pytest.raises(ValueError, match="reference_plane must be finite"):
        sim.add_waveguide_port(0.01, reference_plane=float("nan"))


def test_waveguide_port_rejects_invalid_integer_like_inputs():
    """Offsets, mode, and frequency-count inputs should be integer-safe."""
    sim = Simulation(
        freq_max=10e9,
        domain=(0.12, 0.04, 0.02),
        boundary="cpml",
        cpml_layers=10,
        dx=0.002,
    )
    with pytest.raises(ValueError, match="probe_offset and ref_offset must be positive integers"):
        sim.add_waveguide_port(0.01, probe_offset=1.5)
    with pytest.raises(ValueError, match="probe_offset and ref_offset must be positive integers"):
        sim.add_waveguide_port(0.01, ref_offset=0)
    with pytest.raises(ValueError, match="mode must be a tuple of two integers"):
        sim.add_waveguide_port(0.01, mode=(1.0, 0))
    with pytest.raises(ValueError, match="mode indices must be non-negative"):
        sim.add_waveguide_port(0.01, mode=(-1, 0))
    with pytest.raises(ValueError, match="n_freqs must be a positive integer"):
        sim.add_waveguide_port(0.01, n_freqs=12.5)


def test_waveguide_port_rejects_invalid_freq_arrays():
    """Explicit frequency arrays should be finite, positive, and 1-D."""
    sim = Simulation(
        freq_max=10e9,
        domain=(0.12, 0.04, 0.02),
        boundary="cpml",
        cpml_layers=10,
        dx=0.002,
    )
    with pytest.raises(ValueError, match="freqs must contain only finite values"):
        sim.add_waveguide_port(0.01, freqs=jnp.array([5e9, jnp.nan]))
    with pytest.raises(ValueError, match="freqs must contain only positive values"):
        sim.add_waveguide_port(0.01, freqs=jnp.array([5e9, -1.0]))


def test_waveguide_port_rejects_measurement_planes_outside_domain():
    """Stored ref/probe measurement planes must stay inside the physical domain."""
    sim = Simulation(
        freq_max=10e9,
        domain=(0.03, 0.04, 0.02),
        boundary="cpml",
        cpml_layers=10,
        dx=0.002,
    )
    with pytest.raises(ValueError, match="measurement planes exceed the physical x-domain after grid snapping"):
        sim.add_waveguide_port(
            0.024,
            probe_offset=6,
            ref_offset=3,
        )


def test_periodic_axes_reject_specialized_source_conflicts():
    """Manual periodic overrides should not mix with specialized boundary setups."""
    sim_tfsf = Simulation(
        freq_max=8e9,
        domain=(0.08, 0.006, 0.006),
        boundary="cpml",
        cpml_layers=8,
        dx=0.001,
    )
    sim_tfsf.set_periodic_axes("y")
    with pytest.raises(ValueError, match="periodic-axis overrides"):
        sim_tfsf.add_tfsf_source()

    sim_wg = Simulation(
        freq_max=10e9,
        domain=(0.12, 0.04, 0.02),
        boundary="cpml",
        cpml_layers=10,
        dx=0.002,
    )
    sim_wg.set_periodic_axes("x")
    with pytest.raises(ValueError, match="periodic-axis overrides"):
        sim_wg.add_waveguide_port(0.01)


def test_periodic_axes_through_api_matches_low_level():
    """High-level periodic-axis control should match low-level runner behavior."""
    sim = Simulation(freq_max=5e9, domain=(0.015, 0.015, 0.015), boundary="pec")
    sim.set_periodic_axes("y")
    sim.add_port((0.005, 0.001, 0.0075), "ez")
    sim.add_probe((0.005, 0.014, 0.0075), "ez")

    api_result = sim.run(n_steps=40, compute_s_params=False)

    grid = sim._build_grid()
    materials, _, _ = sim._build_materials(grid)
    port_entry = sim._ports[0]
    port = LumpedPort(
        position=port_entry.position,
        component=port_entry.component,
        impedance=port_entry.impedance,
        excitation=port_entry.waveform,
    )
    materials = setup_lumped_port(grid, port, materials)
    src = make_port_source(grid, port, materials, 40)
    prb = make_probe(grid, sim._probes[0].position, sim._probes[0].component)
    low_result = low_level_run(
        grid,
        materials,
        40,
        boundary="pec",
        periodic=(False, True, False),
        sources=[src],
        probes=[prb],
    )

    diff = float(jnp.max(jnp.abs(api_result.time_series - low_result.time_series)))
    assert diff < 1e-6, f"High-level periodic control diverged from low-level runner: {diff}"


def test_tfsf_requires_supported_configuration():
    """Unsupported TFSF configurations should fail clearly."""
    sim_pec = Simulation(freq_max=8e9, domain=(0.08, 0.006, 0.006), boundary="pec")
    with pytest.raises(ValueError, match="boundary='cpml'"):
        sim_pec.add_tfsf_source()

    sim_no_cpml = Simulation(
        freq_max=8e9,
        domain=(0.08, 0.006, 0.006),
        boundary="cpml",
        cpml_layers=0,
    )
    with pytest.raises(ValueError, match="cpml_layers > 0"):
        sim_no_cpml.add_tfsf_source()

    sim_ports = Simulation(
        freq_max=8e9,
        domain=(0.08, 0.006, 0.006),
        boundary="cpml",
        cpml_layers=8,
        dx=0.001,
    )
    sim_ports.add_tfsf_source()
    with pytest.raises(ValueError, match="Lumped ports|lumped ports"):
        sim_ports.add_port((0.01, 0.003, 0.003), "ez")

    with pytest.raises(ValueError, match="polarization"):
        sim_ports.add_tfsf_source(polarization="ex")
    with pytest.raises(ValueError, match="direction"):
        sim_ports.add_tfsf_source(direction="y")
    with pytest.raises(ValueError, match="abs\\(angle_deg\\)"):
        sim_ports.add_tfsf_source(angle_deg=90.0)


def test_tfsf_rejects_non_vacuum_boundary_buffer():
    """TFSF should fail loudly if geometry touches the TFSF boundary planes."""
    sim = Simulation(
        freq_max=8e9,
        domain=(0.08, 0.006, 0.006),
        boundary="cpml",
        cpml_layers=8,
        dx=0.001,
    )
    sim.add_material("fill", eps_r=4.0)
    sim.add(Box((0.0, 0.0, 0.0), (0.08, 0.006, 0.006)), material="fill")
    sim.add_tfsf_source()

    with pytest.raises(ValueError, match="vacuum"):
        sim.run(n_steps=10, compute_s_params=False)


def test_unknown_material_raises():
    """Unknown material name raises KeyError."""
    sim = Simulation(freq_max=5e9, domain=(0.03, 0.03, 0.03))
    with pytest.raises(KeyError, match="Unknown material"):
        sim.add(Box((0, 0, 0), (0.01, 0.01, 0.01)), material="unobtanium")


def test_validation_errors():
    """Bad inputs raise ValueError."""
    with pytest.raises(ValueError, match="freq_max"):
        Simulation(freq_max=-1e9, domain=(0.03, 0.03, 0.03))

    with pytest.raises(ValueError, match="domain"):
        Simulation(freq_max=5e9, domain=(0.03, -0.03, 0.03))

    with pytest.raises(ValueError, match="boundary"):
        Simulation(freq_max=5e9, domain=(0.03, 0.03, 0.03), boundary="abc")

    with pytest.raises(ValueError, match="boundary='upml'"):
        Simulation(freq_max=5e9, domain=(0.03, 0.03, 0.03), boundary="upml", solver="adi")

    with pytest.raises(ValueError, match="dz_profile"):
        Simulation(
            freq_max=5e9,
            domain=(0.03, 0.03, 0.003),
            boundary="upml",
            dz_profile=np.array([1e-3, 1e-3, 1e-3]),
        )

    sim = Simulation(freq_max=5e9, domain=(0.03, 0.03, 0.03))
    with pytest.raises(ValueError, match="component"):
        sim.add_port((0.01, 0.01, 0.01), "bz")
    with pytest.raises(ValueError, match="impedance"):
        sim.add_port((0.01, 0.01, 0.01), "ez", impedance=-50)
    with pytest.raises(ValueError, match="axis"):
        sim.add_dft_plane_probe(axis="q", coordinate=0.01)
    with pytest.raises(ValueError, match="outside"):
        sim.add_dft_plane_probe(axis="x", coordinate=1.0)
    with pytest.raises(ValueError, match="periodic axes"):
        sim.set_periodic_axes("q")


def test_floquet_auto_mesh_rejects_nonuniform_fallback():
    """Floquet workflows should fail instead of silently dropping auto NU mesh."""
    sim = Simulation(freq_max=5e9, domain=(0.03, 0.03, 0.005), boundary="cpml")
    sim.add_material("sub", eps_r=4.4, sigma=0.025)
    sim.add(Box((0.0, 0.0, 0.0), (0.03, 0.03, 0.0016)), material="sub")
    sim.add_floquet_port(0.0025, axis="z", scan_theta=0.0)

    with pytest.raises(ValueError, match="Floquet ports do not support non-uniform z mesh"):
        sim.run(n_steps=10, compute_s_params=False)


def test_fluent_api():
    """Builder methods return self for chaining."""
    sim = (
        Simulation(freq_max=5e9, domain=(0.03, 0.03, 0.03), boundary="pec")
        .add_material("sub", eps_r=4.0)
        .add(Box((0, 0, 0), (0.03, 0.03, 0.001)), material="sub")
        .add_port((0.015, 0.015, 0.001), "ez")
        .add_probe((0.02, 0.02, 0.015), "ez")
    )
    result = sim.run(n_steps=30, compute_s_params=False)
    assert result.time_series.shape == (30, 1)


def test_checkpoint_through_api():
    """Checkpoint mode works through the high-level API."""
    sim = Simulation(freq_max=5e9, domain=(0.015, 0.015, 0.015), boundary="pec")
    sim.add_port((0.005, 0.0075, 0.0075), "ez")
    sim.add_probe((0.01, 0.0075, 0.0075), "ez")

    r1 = sim.run(n_steps=30, checkpoint=False, compute_s_params=False)
    r2 = sim.run(n_steps=30, checkpoint=True, compute_s_params=False)

    diff = float(jnp.max(jnp.abs(r1.time_series - r2.time_series)))
    assert diff == 0.0, f"Checkpoint changed result: {diff}"


def test_gradient_through_api():
    """AD gradient flows through the high-level API."""
    sim = Simulation(freq_max=5e9, domain=(0.015, 0.015, 0.015), boundary="pec")
    sim.add_probe((0.01, 0.0075, 0.0075), "ez")

    grid = sim._build_grid()

    pulse = GaussianPulse(f0=3e9, bandwidth=0.5)
    from rfx.core.yee import MaterialArrays
    from rfx.simulation import make_source, make_probe, run

    src = make_source(grid, (0.005, 0.0075, 0.0075), "ez", pulse, 30)
    prb = make_probe(grid, (0.01, 0.0075, 0.0075), "ez")

    def objective(eps_r):
        sigma = jnp.zeros(grid.shape, dtype=jnp.float32)
        mu_r = jnp.ones(grid.shape, dtype=jnp.float32)
        mats = MaterialArrays(eps_r=eps_r, sigma=sigma, mu_r=mu_r)
        result = run(grid, mats, 30, sources=[src], probes=[prb], checkpoint=True)
        return jnp.sum(result.time_series ** 2)

    eps_r = jnp.ones(grid.shape, dtype=jnp.float32)
    grad = jax.grad(objective)(eps_r)
    assert float(jnp.max(jnp.abs(grad))) > 1e-15, "Gradient is zero"


def test_repr():
    """repr() gives useful summary."""
    sim = Simulation(freq_max=5e9, domain=(0.03, 0.03, 0.03))
    sim.add_material("sub", eps_r=4.0)
    sim.add(Box((0, 0, 0), (0.01, 0.01, 0.01)), material="sub")
    sim.add_port((0.015, 0.015, 0.015), "ez")
    sim.add_probe((0.02, 0.02, 0.02), "ez")

    s = repr(sim)
    assert "5.00e+09" in s
    assert "ports=1" in s
    assert "probes=1" in s


def test_material_library_has_common_entries():
    """Material library contains expected RF materials."""
    for name in ["vacuum", "fr4", "copper", "aluminum", "ptfe", "alumina"]:
        assert name in MATERIAL_LIBRARY, f"Missing library material: {name}"


def test_sparams_respond_to_lorentz_dispersion():
    """API S-parameter extraction should reflect Lorentz materials."""
    freqs = np.linspace(1e9, 4e9, 5)

    sim_plain = Simulation(freq_max=5e9, domain=(0.01, 0.01, 0.01), boundary="pec")
    sim_plain.add_material("mat", eps_r=2.0)
    sim_plain.add(Box((0.0, 0.0, 0.0), (0.01, 0.01, 0.01)), material="mat")
    sim_plain.add_port((0.002, 0.005, 0.005), "ez")
    sim_plain.add_port((0.008, 0.005, 0.005), "ez")

    sim_lorentz = Simulation(freq_max=5e9, domain=(0.01, 0.01, 0.01), boundary="pec")
    sim_lorentz.add_material(
        "mat",
        eps_r=2.0,
        lorentz_poles=[lorentz_pole(2.0, 2 * np.pi * 3e9, 1e8)],
    )
    sim_lorentz.add(Box((0.0, 0.0, 0.0), (0.01, 0.01, 0.01)), material="mat")
    sim_lorentz.add_port((0.002, 0.005, 0.005), "ez")
    sim_lorentz.add_port((0.008, 0.005, 0.005), "ez")

    plain = sim_plain.run(
        n_steps=20,
        compute_s_params=True,
        s_param_freqs=freqs,
        s_param_n_steps=20,
    )
    lorentz = sim_lorentz.run(
        n_steps=20,
        compute_s_params=True,
        s_param_freqs=freqs,
        s_param_n_steps=20,
    )

    assert plain.s_params is not None
    assert lorentz.s_params is not None
    assert np.max(np.abs(plain.s_params - lorentz.s_params)) > 1e-6


def test_five_line_patch_workflow():
    """Verify the minimal patch antenna workflow runs in <= 5 API calls.

    Five user-facing lines (excluding import):
      1. sim = Simulation(freq_max=..., domain=..., boundary="cpml")
      2. sim.add(patch, material="pec")
      3. sim.add(substrate, material="fr4")
      4. sim.add_port(position=..., component="ez")
      5. result = sim.run(n_steps=...)

    Uses a coarse grid (dx=5mm) and few steps for fast CI execution.
    """
    sim = rfx.Simulation(
        freq_max=4e9, domain=(0.08, 0.06, 0.02),
        boundary="cpml", cpml_layers=8, dx=5e-3,
    )
    sim.add(
        Box((-19e-3, -14.5e-3, 0.8e-3), (19e-3, 14.5e-3, 0.8e-3)),
        material="pec",
    )  # patch
    sim.add(
        Box((-30e-3, -25e-3, 0), (30e-3, 25e-3, 1.6e-3)),
        material="fr4",
    )  # substrate
    sim.add_port(position=(5e-3, 0, 0.8e-3), component="ez")
    result = sim.run(n_steps=50)

    assert isinstance(result, Result)
    # The port excites the domain; verify non-zero fields in final state
    ez_peak = float(jnp.max(jnp.abs(result.state.ez)))
    assert ez_peak > 0, "Port should excite non-zero Ez field"
