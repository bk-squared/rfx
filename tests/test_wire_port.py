"""Tests for multi-cell (wire) lumped port."""

import numpy as np

from rfx.grid import Grid
from rfx.core.yee import init_state, init_materials, update_e, update_h
from rfx.sources.sources import (
    GaussianPulse, WirePort, setup_wire_port, apply_wire_port,
)


def test_wire_port_distributed_impedance():
    """sigma should be modified at all cells along the wire."""
    grid = Grid(freq_max=5e9, domain=(0.01, 0.01, 0.01), dx=0.001, cpml_layers=0)
    materials = init_materials(grid.shape)

    port = WirePort(
        start=(0.005, 0.005, 0.002),
        end=(0.005, 0.005, 0.008),
        component="ez",
        impedance=50.0,
        excitation=GaussianPulse(f0=3e9),
    )

    updated = setup_wire_port(grid, port, materials)
    sigma = np.array(updated.sigma)

    # Find modified cells
    modified = np.argwhere(sigma > 0)
    print("\nWire port distributed impedance:")
    print(f"  Modified cells: {len(modified)}")
    print(f"  Max sigma: {sigma.max():.2e}")

    assert len(modified) >= 3, f"Wire should span multiple cells, got {len(modified)}"
    # All modified cells should be along z-axis at the wire position
    assert np.all(modified[:, 0] == modified[0, 0]), "All cells same x"
    assert np.all(modified[:, 1] == modified[0, 1]), "All cells same y"


def test_wire_port_excites_vertical_field():
    """Ez should be non-zero across multiple z-cells after excitation."""
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
    # Run a few steps
    for n in range(20):
        t = n * grid.dt
        state = update_h(state, materials, grid.dt, grid.dx)
        state = update_e(state, materials, grid.dt, grid.dx)
        state = apply_wire_port(state, grid, port, t, materials)

    # Check Ez along the wire
    ix = grid.position_to_index((0.01, 0.01, 0.005))[0]
    iy = grid.position_to_index((0.01, 0.01, 0.005))[1]
    ez_along_z = np.array(state.ez[ix, iy, :])
    n_excited = int(np.sum(np.abs(ez_along_z) > 1e-10))

    print("\nVertical excitation:")
    print(f"  Ez non-zero z-cells: {n_excited}")
    print(f"  Max |Ez|: {np.max(np.abs(ez_along_z)):.4e}")

    assert n_excited >= 3, f"Wire should excite multiple z-cells, got {n_excited}"


def test_wire_port_cavity_resonance():
    """Wire port in a PEC cavity should excite a resonance."""
    # Simple PEC box cavity with wire port — no substrate complexity
    grid = Grid(freq_max=8e9, domain=(0.03, 0.03, 0.01), dx=0.001, cpml_layers=0)
    materials = init_materials(grid.shape)

    # Wire port spanning z: vertical probe in the cavity
    port = WirePort(
        start=(0.01, 0.015, 0.002),
        end=(0.01, 0.015, 0.008),
        component="ez",
        impedance=50.0,
        excitation=GaussianPulse(f0=5e9, bandwidth=0.8),
    )
    materials = setup_wire_port(grid, port, materials)

    state = init_state(grid.shape)
    from rfx.boundaries.pec import apply_pec

    n_steps = 2000
    feed_idx = grid.position_to_index((0.01, 0.015, 0.005))
    ez_trace = np.zeros(n_steps)

    for n in range(n_steps):
        t = n * grid.dt
        state = update_h(state, materials, grid.dt, grid.dx)
        state = update_e(state, materials, grid.dt, grid.dx)
        state = apply_pec(state)
        state = apply_wire_port(state, grid, port, t, materials)
        ez_trace[n] = float(state.ez[feed_idx])

    # FFT
    spec = np.abs(np.fft.rfft(ez_trace, n=len(ez_trace) * 4))
    freqs = np.fft.rfftfreq(len(ez_trace) * 4, d=grid.dt)
    band = (freqs > 2e9) & (freqs < 8e9)
    peak_f = freqs[band][np.argmax(spec[band])]

    print("\nWire port cavity resonance:")
    print(f"  Peak freq: {peak_f/1e9:.2f} GHz")
    print(f"  Max |Ez|:  {np.max(np.abs(ez_trace)):.4e}")

    assert np.max(np.abs(ez_trace)) > 1e-6, "Wire port should excite fields"
    assert not np.any(np.isnan(ez_trace)), "No NaN allowed"
    assert 2e9 < peak_f < 8e9, f"Peak {peak_f/1e9:.1f} GHz out of range"


def test_wire_port_api_extent():
    """High-level API: add_port with extent creates a WirePort under the hood."""
    from rfx.api import Simulation
    from rfx.sources.sources import GaussianPulse

    sim = Simulation(freq_max=5e9, domain=(0.02, 0.02, 0.01), dx=0.001)
    sim.add_port(
        position=(0.01, 0.01, 0.002),
        component="ez",
        impedance=50.0,
        waveform=GaussianPulse(f0=3e9),
        extent=0.006,  # span from z=0.002 to z=0.008
    )
    sim.add_probe((0.01, 0.01, 0.005), "ez")
    result = sim.run(n_steps=200)

    # The probe should record non-zero field
    ts = np.array(result.time_series[:, 0])
    assert np.max(np.abs(ts)) > 1e-10, "Wire port via API should excite fields"
    assert not np.any(np.isnan(ts)), "No NaN in time series"
    print("\nAPI wire port test:")
    print(f"  Max |Ez| at probe: {np.max(np.abs(ts)):.4e}")


def test_wire_port_live_cell_split_and_all_dead_guard():
    """Issue #318: setup_wire_port distributes sigma over LIVE cells only.

    - pec_mask=None (or all-live) is bit-identical to the historical
      all-cells fold (the structural byte-identity predicate).
    - A dead extent cell (pec_mask True) gets NO port sigma and the live
      cells carry sigma scaled by n_live (per-cell R = Z0/n_live, series
      sum over live cells = Z0 exactly).
    - All extent cells dead -> ValueError (no live cell to terminate).
    """
    import jax.numpy as jnp
    import pytest

    grid = Grid(freq_max=5e9, domain=(0.01, 0.01, 0.01), dx=0.001,
                cpml_layers=0)
    materials = init_materials(grid.shape)
    port = WirePort(
        start=(0.005, 0.005, 0.002),
        end=(0.005, 0.005, 0.004),
        component="ez",
        impedance=50.0,
        excitation=GaussianPulse(f0=3e9),
    )
    from rfx.sources.sources import _wire_port_cells
    cells = _wire_port_cells(grid, port)
    assert len(cells) == 3

    # pec_mask=None == all-live mask, bit-identical (structural predicate)
    legacy = np.asarray(setup_wire_port(grid, port, materials).sigma)
    all_live = np.asarray(setup_wire_port(
        grid, port, materials,
        pec_mask=jnp.zeros(grid.shape, dtype=jnp.bool_)).sigma)
    assert np.array_equal(legacy, all_live), (
        "all-live pec_mask must be bit-identical to pec_mask=None")

    # One dead cell: sigma only at live cells, series R over live == Z0
    mask = jnp.zeros(grid.shape, dtype=jnp.bool_)
    top = cells[-1]
    mask = mask.at[top[0], top[1], top[2]].set(True)
    folded = np.asarray(setup_wire_port(grid, port, materials,
                                        pec_mask=mask).sigma)
    assert folded[top[0], top[1], top[2]] == 0.0, (
        "dead extent cell must carry no port sigma")
    r_series = 0.0
    for c in cells[:-1]:
        s = float(folded[c[0], c[1], c[2]])
        assert s > 0.0
        r_series += 1.0 / (s * grid.dx)
    assert abs(r_series - 50.0) < 1e-9, (
        f"series termination over live cells must equal Z0=50, got {r_series}")

    # All cells dead -> loud ValueError
    mask_all = jnp.zeros(grid.shape, dtype=jnp.bool_)
    for c in cells:
        mask_all = mask_all.at[c[0], c[1], c[2]].set(True)
    with pytest.raises(ValueError, match="no live cell"):
        setup_wire_port(grid, port, materials, pec_mask=mask_all)
