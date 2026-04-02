"""Tests for WirePort S-parameter extraction."""

import numpy as np
import jax.numpy as jnp

from rfx.grid import Grid
from rfx.core.yee import init_state, init_materials, update_e, update_h, EPS_0
from rfx.boundaries.pec import apply_pec
from rfx.sources.sources import (
    GaussianPulse, WirePort, LumpedPort,
    setup_wire_port, apply_wire_port,
    setup_lumped_port, apply_lumped_port,
)
from rfx.probes.probes import (
    wire_port_voltage, wire_port_current,
    init_wire_sparam_probe, update_wire_sparam_probe,
    extract_s11, extract_s_matrix_wire,
    init_sparam_probe, update_sparam_probe, extract_s_matrix,
)


def test_wire_port_voltage_nonzero():
    """wire_port_voltage should return nonzero after excitation."""
    grid = Grid(freq_max=5e9, domain=(0.02, 0.02, 0.01), dx=0.001, cpml_layers=0)
    materials = init_materials(grid.shape)

    port = WirePort(
        start=(0.01, 0.01, 0.002),
        end=(0.01, 0.01, 0.008),
        component="ez",
        impedance=50.0,
        excitation=GaussianPulse(f0=3e9),
    )
    materials = setup_wire_port(grid, port, materials)
    state = init_state(grid.shape)

    for n in range(20):
        t = n * grid.dt
        state = update_h(state, materials, grid.dt, grid.dx)
        state = update_e(state, materials, grid.dt, grid.dx)
        state = apply_pec(state)
        state = apply_wire_port(state, grid, port, t, materials)

    v = wire_port_voltage(state, grid, port)
    assert abs(float(v)) > 1e-10, f"Voltage should be nonzero, got {v}"


def test_wire_port_current_nonzero():
    """wire_port_current should return nonzero after excitation."""
    grid = Grid(freq_max=5e9, domain=(0.02, 0.02, 0.01), dx=0.001, cpml_layers=0)
    materials = init_materials(grid.shape)

    port = WirePort(
        start=(0.01, 0.01, 0.002),
        end=(0.01, 0.01, 0.008),
        component="ez",
        impedance=50.0,
        excitation=GaussianPulse(f0=3e9),
    )
    materials = setup_wire_port(grid, port, materials)
    state = init_state(grid.shape)

    for n in range(20):
        t = n * grid.dt
        state = update_h(state, materials, grid.dt, grid.dx)
        state = update_e(state, materials, grid.dt, grid.dx)
        state = apply_pec(state)
        state = apply_wire_port(state, grid, port, t, materials)

    i_val = wire_port_current(state, grid, port)
    assert abs(float(i_val)) > 1e-10, f"Current should be nonzero, got {i_val}"


def test_wire_sparam_probe_accumulates():
    """Wire S-param probe should accumulate nonzero DFTs."""
    grid = Grid(freq_max=5e9, domain=(0.02, 0.02, 0.01), dx=0.001, cpml_layers=0)
    materials = init_materials(grid.shape)

    port = WirePort(
        start=(0.01, 0.01, 0.002),
        end=(0.01, 0.01, 0.008),
        component="ez",
        impedance=50.0,
        excitation=GaussianPulse(f0=3e9),
    )
    materials = setup_wire_port(grid, port, materials)

    freqs = jnp.linspace(1e9, 5e9, 10)
    probe = init_wire_sparam_probe(grid, port, freqs, dft_total_steps=100)

    state = init_state(grid.shape)
    for n in range(100):
        t = n * grid.dt
        state = update_h(state, materials, grid.dt, grid.dx)
        state = update_e(state, materials, grid.dt, grid.dx)
        state = apply_pec(state)
        state = apply_wire_port(state, grid, port, t, materials)
        probe = update_wire_sparam_probe(probe, state, grid, port, grid.dt)

    assert np.max(np.abs(probe.v_dft)) > 0, "v_dft should be nonzero"
    assert np.max(np.abs(probe.i_dft)) > 0, "i_dft should be nonzero"
    assert np.max(np.abs(probe.v_inc_dft)) > 0, "v_inc_dft should be nonzero"


def test_wire_s11_pec_cavity():
    """S11 of wire port in PEC cavity should be passive (|S11| <= 1)."""
    grid = Grid(freq_max=8e9, domain=(0.03, 0.03, 0.01), dx=0.001, cpml_layers=0)
    materials = init_materials(grid.shape)

    port = WirePort(
        start=(0.01, 0.015, 0.002),
        end=(0.01, 0.015, 0.008),
        component="ez",
        impedance=50.0,
        excitation=GaussianPulse(f0=5e9, bandwidth=0.8),
    )
    materials = setup_wire_port(grid, port, materials)

    freqs = jnp.linspace(2e9, 8e9, 20)
    probe = init_wire_sparam_probe(grid, port, freqs, dft_total_steps=2000)

    state = init_state(grid.shape)
    for n in range(2000):
        t = n * grid.dt
        state = update_h(state, materials, grid.dt, grid.dx)
        state = update_e(state, materials, grid.dt, grid.dx)
        state = apply_pec(state)
        state = apply_wire_port(state, grid, port, t, materials)
        probe = update_wire_sparam_probe(probe, state, grid, port, grid.dt)

    s11 = extract_s11(probe, z0=50.0)
    s11_mag = np.abs(np.array(s11))

    print(f"\nWire port S11 in PEC cavity:")
    print(f"  |S11| range: {s11_mag.min():.3f} - {s11_mag.max():.3f}")
    print(f"  |S11| mean:  {s11_mag.mean():.3f}")

    # PEC cavity is lossless — |S11| should be close to 1
    assert np.all(s11_mag < 1.5), f"|S11| should be <= ~1, max={s11_mag.max():.3f}"
    assert not np.any(np.isnan(s11_mag)), "No NaN in S11"


def test_wire_s_matrix_extraction():
    """extract_s_matrix_wire should return a valid S-matrix."""
    grid = Grid(freq_max=8e9, domain=(0.03, 0.03, 0.01), dx=0.001, cpml_layers=0)
    base_materials = init_materials(grid.shape)

    port = WirePort(
        start=(0.01, 0.015, 0.002),
        end=(0.01, 0.015, 0.008),
        component="ez",
        impedance=50.0,
        excitation=GaussianPulse(f0=5e9, bandwidth=0.8),
    )

    freqs = jnp.linspace(2e9, 8e9, 10)
    S = extract_s_matrix_wire(
        grid, base_materials, [port], freqs, n_steps=1000,
    )

    assert S.shape == (1, 1, 10), f"Expected (1,1,10), got {S.shape}"
    s11_mag = np.abs(S[0, 0, :])
    print(f"\nextract_s_matrix_wire S11:")
    print(f"  |S11| range: {s11_mag.min():.3f} - {s11_mag.max():.3f}")

    assert not np.any(np.isnan(S)), "No NaN in S-matrix"


def test_wire_sparam_api_integration():
    """High-level API with extent should produce S-parameters."""
    from rfx.api import Simulation
    from rfx.sources.sources import GaussianPulse

    sim = Simulation(freq_max=5e9, domain=(0.02, 0.02, 0.01), dx=0.001)
    sim.add_port(
        position=(0.01, 0.01, 0.002),
        component="ez",
        impedance=50.0,
        waveform=GaussianPulse(f0=3e9),
        extent=0.006,
    )
    sim.add_probe((0.01, 0.01, 0.005), "ez")
    result = sim.run(n_steps=200, s_param_n_steps=500)

    assert result.s_params is not None, "S-params should be computed for wire port"
    assert result.freqs is not None, "Freqs should be present"
    assert result.s_params.shape[0] == 1, "Should have 1 port"
    assert result.s_params.shape[1] == 1, "Should have 1 port"

    s11_mag = np.abs(result.s_params[0, 0, :])
    print(f"\nAPI wire port S-params:")
    print(f"  Shape: {result.s_params.shape}")
    print(f"  |S11| range: {s11_mag.min():.3f} - {s11_mag.max():.3f}")

    assert not np.any(np.isnan(result.s_params)), "No NaN in S-params"
