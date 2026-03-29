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
from rfx.api import Simulation, Result, MATERIAL_LIBRARY, MaterialSpec
from rfx.geometry.csg import Box, Sphere
from rfx.sources.sources import GaussianPulse, LumpedPort, setup_lumped_port
from rfx.sources.waveguide_port import extract_waveguide_s21
from rfx.materials.debye import DebyePole
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
