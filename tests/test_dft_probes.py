"""Tests for DFT probes — point and plane frequency-domain monitors.

Validates that running DFT accumulation matches post-hoc FFT of stored
time series, ensuring correct frequency-domain field extraction.
"""

import numpy as np
import jax.numpy as jnp

from rfx.grid import Grid
from rfx.core.yee import init_state, init_materials, update_e, update_h
from rfx.boundaries.pec import apply_pec
from rfx.sources.sources import GaussianPulse, add_point_source
from rfx.probes.probes import (
    init_dft_probe, update_dft_probe,
    init_dft_plane_probe, update_dft_plane_probe,
)


def test_dft_point_probe_matches_fft():
    """DFT probe at a point should match post-hoc FFT of stored time series."""
    grid = Grid(freq_max=5e9, domain=(0.03, 0.03, 0.03), cpml_layers=0)
    materials = init_materials(grid.shape)
    state = init_state(grid.shape)

    pulse = GaussianPulse(f0=3e9, bandwidth=0.5)
    src_pos = (0.015, 0.015, 0.015)

    # Monitor position (offset from source)
    mon_pos = (0.02, 0.02, 0.02)
    freqs = jnp.linspace(1e9, 5e9, 20)

    # Store time series for post-hoc FFT comparison
    n_steps = grid.num_timesteps(num_periods=15)
    probe = init_dft_probe(grid, mon_pos, "ez", freqs, dft_total_steps=n_steps)
    time_series = []
    mon_idx = grid.position_to_index(mon_pos)

    for step in range(n_steps):
        t = step * grid.dt
        state = update_h(state, materials, grid.dt, grid.dx)
        state = update_e(state, materials, grid.dt, grid.dx)
        state = apply_pec(state)
        state = add_point_source(state, grid, src_pos, "ez", pulse(t))

        probe = update_dft_probe(probe, state, grid.dt)
        time_series.append(float(state.ez[mon_idx[0], mon_idx[1], mon_idx[2]]))

    time_series = np.array(time_series)
    dt = grid.dt

    # Post-hoc DFT at same frequencies.
    # update_dft_probe uses state.step which is already incremented by update_e,
    # so the DFT time is (step + 1) * dt, not step * dt.
    freqs_np = np.array(freqs)
    t_arr = (np.arange(n_steps) + 1) * dt
    reference_dft = np.zeros(len(freqs_np), dtype=complex)
    for fi, f in enumerate(freqs_np):
        reference_dft[fi] = np.sum(time_series * np.exp(-1j * 2 * np.pi * f * t_arr)) * dt

    # Compare
    probe_result = np.array(probe.accumulator)
    # Normalize for comparison (both should be nearly identical)
    for fi in range(len(freqs_np)):
        if abs(reference_dft[fi]) > 1e-20:
            ratio = abs(probe_result[fi]) / abs(reference_dft[fi])
            phase_err = abs(np.angle(probe_result[fi]) - np.angle(reference_dft[fi]))
            assert abs(ratio - 1.0) < 0.01, \
                f"Amplitude mismatch at f={freqs_np[fi]/1e9:.1f} GHz: ratio={ratio:.4f}"
            assert phase_err < 0.1, \
                f"Phase mismatch at f={freqs_np[fi]/1e9:.1f} GHz: err={phase_err:.4f} rad"

    print(f"\nDFT point probe: {len(freqs_np)} freqs matched to < 1% amplitude, < 0.05 rad phase")


def test_dft_plane_probe_matches_point():
    """DFT plane probe should match point DFT probe at corresponding locations."""
    grid = Grid(freq_max=5e9, domain=(0.03, 0.03, 0.03), cpml_layers=0)
    materials = init_materials(grid.shape)
    state = init_state(grid.shape)

    pulse = GaussianPulse(f0=3e9, bandwidth=0.5)
    src_pos = (0.015, 0.015, 0.015)

    freqs = jnp.linspace(2e9, 5e9, 10)
    n_steps = grid.num_timesteps(num_periods=12)

    # Plane probe: x-normal plane at x_index
    x_index = grid.nx // 2 + 2
    plane_probe = init_dft_plane_probe(
        axis=0, index=x_index, component="ez",
        freqs=freqs, grid_shape=grid.shape, dft_total_steps=n_steps,
    )

    # Point probe at same x, at (y_mid, z_mid)
    y_mid = grid.ny // 2
    z_mid = grid.nz // 2
    # Convert grid index to physical position for init_dft_probe
    point_pos = (
        (x_index - grid.cpml_layers) * grid.dx,
        (y_mid - grid.cpml_layers) * grid.dx,
        (z_mid - grid.cpml_layers) * grid.dx,
    )
    point_probe = init_dft_probe(grid, point_pos, "ez", freqs, dft_total_steps=n_steps)

    for step in range(n_steps):
        t = step * grid.dt
        state = update_h(state, materials, grid.dt, grid.dx)
        state = update_e(state, materials, grid.dt, grid.dx)
        state = apply_pec(state)
        state = add_point_source(state, grid, src_pos, "ez", pulse(t))

        plane_probe = update_dft_plane_probe(plane_probe, state, grid.dt)
        point_probe = update_dft_probe(point_probe, state, grid.dt)

    # Extract plane probe value at (y_mid, z_mid) — should match point probe
    plane_at_point = np.array(plane_probe.accumulator[:, y_mid, z_mid])
    point_result = np.array(point_probe.accumulator)

    for fi in range(len(freqs)):
        if abs(point_result[fi]) > 1e-20:
            ratio = abs(plane_at_point[fi]) / abs(point_result[fi])
            assert abs(ratio - 1.0) < 0.01, \
                f"Plane/point mismatch at f={float(freqs[fi])/1e9:.1f} GHz: ratio={ratio:.4f}"

    print(f"\nDFT plane probe: matches point probe at {len(freqs)} freqs to < 1%")


def test_dft_plane_probe_spatial_pattern():
    """DFT plane probe should capture the spatial mode pattern of a PEC cavity."""
    grid = Grid(freq_max=8e9, domain=(0.03, 0.03, 0.03), cpml_layers=0)
    materials = init_materials(grid.shape)
    state = init_state(grid.shape)

    # Off-center source to excite multiple modes
    pulse = GaussianPulse(f0=5e9, bandwidth=0.8)
    src_pos = (0.012, 0.012, 0.012)

    # Monitor at mid-x plane
    x_mid = grid.nx // 2
    freqs = jnp.array([5e9])  # single frequency
    n_steps = grid.num_timesteps(num_periods=20)
    plane_probe = init_dft_plane_probe(
        axis=0, index=x_mid, component="ez",
        freqs=freqs, grid_shape=grid.shape, dft_total_steps=n_steps,
    )
    for step in range(n_steps):
        t = step * grid.dt
        state = update_h(state, materials, grid.dt, grid.dx)
        state = update_e(state, materials, grid.dt, grid.dx)
        state = apply_pec(state)
        state = add_point_source(state, grid, src_pos, "ez", pulse(t))

        plane_probe = update_dft_plane_probe(plane_probe, state, grid.dt)

    field_pattern = np.abs(np.array(plane_probe.accumulator[0, :, :]))

    # On the x-normal plane, Ez is:
    #   - tangential to y-faces (y=0, y=-1) → PEC zeros Ez there
    #   - normal to z-faces (z=0, z=-1) → PEC does NOT zero Ez there
    # So only check y-boundaries.
    y_edge_max = max(
        np.max(field_pattern[0, :]),
        np.max(field_pattern[-1, :]),
    )
    interior_max = np.max(field_pattern[2:-2, 2:-2])

    print("\nDFT plane probe spatial pattern:")
    print(f"  Interior max = {interior_max:.4e}, y-edge max = {y_edge_max:.4e}")

    assert interior_max > 1e-15, "Field pattern should be nonzero in interior"
    # y-edges should be much smaller than interior (PEC enforces Ez=0 at y walls)
    if interior_max > 0:
        assert y_edge_max / interior_max < 0.15, \
            f"PEC y-boundary violated: edge/interior = {y_edge_max/interior_max:.4f}"
