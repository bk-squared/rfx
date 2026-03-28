"""Tests for the agent-friendly high-level API (Stage 4).

Validates that:
1. Simulation builder creates valid grids and materials
2. Named materials and library materials work
3. Geometry rasterization through the builder works
4. Port + probe simulation produces non-zero results
5. S-parameter extraction through the API works
6. Differentiable mode (checkpoint) works through the API
7. Validation catches bad inputs
"""

import numpy as np
import jax
import jax.numpy as jnp
import pytest

import rfx
from rfx.api import Simulation, Result, MATERIAL_LIBRARY, MaterialSpec
from rfx.geometry.csg import Box, Sphere
from rfx.sources.sources import GaussianPulse
from rfx.materials.debye import DebyePole
from rfx.materials.lorentz import lorentz_pole


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
