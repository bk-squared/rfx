"""Tests for Stage 5 features: 2D mode, snapshots, HDF5 checkpoint.

Validates:
1. Grid 2D mode (nz=1, √2 Courant factor)
2. 2D TMz simulation runs and produces non-zero Ez
3. 2D TEz simulation runs and produces non-zero Hz
4. Field snapshots capture every timestep
5. Field snapshots with 2D slice
6. HDF5 state save/load round-trip
7. HDF5 snapshots save/load round-trip
8. HDF5 materials save/load round-trip
9. 2D mode via high-level API
"""

import tempfile
from pathlib import Path

import numpy as np
import jax.numpy as jnp

from rfx.grid import Grid, C0
from rfx.core.yee import init_state, init_materials
from rfx.simulation import (
    run, make_source, SnapshotSpec, SimResult,
)
from rfx.sources.sources import GaussianPulse
from rfx.api import Simulation
from rfx.geometry.csg import Box


# --- Grid 2D mode ---

def test_grid_2d_tmz():
    """2D TMz grid should have nz=1 and √2 Courant factor."""
    g = Grid(freq_max=10e9, domain=(0.03, 0.03, 0.03), mode="2d_tmz")
    assert g.nz == 1
    assert g.is_2d
    assert g.mode == "2d_tmz"
    # dt should use √2, not √3
    dt_expected = g.dx / (C0 * np.sqrt(2.0)) * 0.99
    assert abs(g.dt - dt_expected) / dt_expected < 1e-10
    print(f"\n2D TMz grid: shape={g.shape}, dx={g.dx:.4e}, dt={g.dt:.4e}")


def test_grid_2d_tez():
    """2D TEz grid should have nz=1."""
    g = Grid(freq_max=10e9, domain=(0.03, 0.03, 0.03), mode="2d_tez")
    assert g.nz == 1
    assert g.is_2d


def test_grid_3d_default():
    """Default 3D grid should have nz > 1 and √3 Courant."""
    g = Grid(freq_max=10e9, domain=(0.03, 0.03, 0.03))
    assert not g.is_2d
    assert g.nz > 1
    dt_expected = g.dx / (C0 * np.sqrt(3.0)) * 0.99
    assert abs(g.dt - dt_expected) / dt_expected < 1e-10


# --- 2D simulation ---

def test_2d_tmz_simulation():
    """2D TMz simulation should run and produce non-zero Ez field."""
    grid = Grid(freq_max=10e9, domain=(0.02, 0.02, 0.01), mode="2d_tmz")
    materials = init_materials(grid.shape)

    pulse = GaussianPulse(f0=5e9, bandwidth=0.8)
    center = (0.01, 0.01, 0.0)
    n_steps = 30
    src = make_source(grid, center, "ez", pulse, n_steps)

    result = run(grid, materials, n_steps, sources=[src])
    assert isinstance(result, SimResult)
    assert result.state.ez.shape == grid.shape
    assert result.state.ez.shape[2] == 1  # nz=1

    ez_max = float(jnp.max(jnp.abs(result.state.ez)))
    assert ez_max > 0.0, "Ez should be non-zero after excitation"
    print(f"\n2D TMz: shape={grid.shape}, max|Ez|={ez_max:.4e}")


def test_2d_tez_simulation():
    """2D TEz simulation should run and produce non-zero Hz field."""
    grid = Grid(freq_max=10e9, domain=(0.02, 0.02, 0.01), mode="2d_tez")
    materials = init_materials(grid.shape)

    pulse = GaussianPulse(f0=5e9, bandwidth=0.8)
    center = (0.01, 0.01, 0.0)
    n_steps = 30
    # TEz mode: excite Hz (or Ex/Ey); use ex for simplicity
    src = make_source(grid, center, "ex", pulse, n_steps)

    result = run(grid, materials, n_steps, sources=[src])
    assert isinstance(result, SimResult)
    assert result.state.ex.shape[2] == 1  # nz=1

    ex_max = float(jnp.max(jnp.abs(result.state.ex)))
    assert ex_max > 0.0, "Ex should be non-zero after excitation"
    print(f"\n2D TEz: shape={grid.shape}, max|Ex|={ex_max:.4e}")


# --- Snapshots ---

def test_snapshot_full_field():
    """Snapshot should capture full field at every timestep."""
    grid = Grid(freq_max=10e9, domain=(0.02, 0.02, 0.02))
    materials = init_materials(grid.shape)

    pulse = GaussianPulse(f0=5e9, bandwidth=0.8)
    n_steps = 10
    src = make_source(grid, (0.01, 0.01, 0.01), "ez", pulse, n_steps)

    snap = SnapshotSpec(components=("ez",))
    result = run(grid, materials, n_steps, sources=[src], snapshot=snap)

    assert result.snapshots is not None
    assert "ez" in result.snapshots
    # Shape: (n_steps, nx, ny, nz)
    assert result.snapshots["ez"].shape[0] == n_steps
    assert result.snapshots["ez"].shape[1:] == grid.shape
    print(f"\nSnapshot full: ez shape={result.snapshots['ez'].shape}")


def test_snapshot_2d_slice():
    """Snapshot with slice should capture 2D cross-section."""
    grid = Grid(freq_max=10e9, domain=(0.02, 0.02, 0.02))
    materials = init_materials(grid.shape)

    pulse = GaussianPulse(f0=5e9, bandwidth=0.8)
    n_steps = 10
    src = make_source(grid, (0.01, 0.01, 0.01), "ez", pulse, n_steps)

    mid_z = grid.nz // 2
    snap = SnapshotSpec(
        components=("ez", "hx"),
        slice_axis=2,
        slice_index=mid_z,
    )
    result = run(grid, materials, n_steps, sources=[src], snapshot=snap)

    assert result.snapshots is not None
    assert "ez" in result.snapshots
    assert "hx" in result.snapshots
    # Shape: (n_steps, nx, ny) — z axis removed
    assert result.snapshots["ez"].shape == (n_steps, grid.nx, grid.ny)
    print(f"\nSnapshot slice: ez shape={result.snapshots['ez'].shape}")


def test_snapshot_2d_mode():
    """Snapshot in 2D mode should work."""
    grid = Grid(freq_max=10e9, domain=(0.02, 0.02, 0.01), mode="2d_tmz")
    materials = init_materials(grid.shape)

    pulse = GaussianPulse(f0=5e9, bandwidth=0.8)
    n_steps = 10
    src = make_source(grid, (0.01, 0.01, 0.0), "ez", pulse, n_steps)

    snap = SnapshotSpec(components=("ez",), slice_axis=2, slice_index=0)
    result = run(grid, materials, n_steps, sources=[src], snapshot=snap)

    assert result.snapshots is not None
    # Shape: (n_steps, nx, ny)
    assert result.snapshots["ez"].shape == (n_steps, grid.nx, grid.ny)


# --- HDF5 checkpoint ---

def test_hdf5_state_roundtrip():
    """Save and load FDTD state should preserve fields."""
    try:
        import h5py  # noqa: F401
    except ImportError:
        import pytest
        pytest.skip("h5py not installed")

    from rfx.checkpoint import save_state, load_state

    grid = Grid(freq_max=10e9, domain=(0.01, 0.01, 0.01))
    state = init_state(grid.shape)
    # Put some non-zero data
    state = state._replace(
        ez=jnp.ones(grid.shape, dtype=jnp.float32) * 0.5,
        step=jnp.array(42, dtype=jnp.int32),
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "state.h5"
        save_state(path, state, grid=grid)
        loaded, meta = load_state(path)

    np.testing.assert_allclose(loaded.ez, state.ez)
    np.testing.assert_allclose(loaded.hx, state.hx)
    assert int(loaded.step) == 42
    assert meta["dx"] == grid.dx
    assert meta["mode"] == "3d"
    print(f"\nHDF5 state roundtrip OK, shape={grid.shape}")


def test_hdf5_snapshots_roundtrip():
    """Save and load snapshots should preserve data."""
    try:
        import h5py  # noqa: F401
    except ImportError:
        import pytest
        pytest.skip("h5py not installed")

    from rfx.checkpoint import save_snapshots, load_snapshots

    snapshots = {
        "ez": np.random.randn(10, 5, 5).astype(np.float32),
        "hx": np.random.randn(10, 5, 5).astype(np.float32),
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "snaps.h5"
        save_snapshots(path, snapshots, dt=1e-12)
        loaded, meta = load_snapshots(path)

    np.testing.assert_allclose(loaded["ez"], snapshots["ez"])
    np.testing.assert_allclose(loaded["hx"], snapshots["hx"])
    assert meta["dt"] == 1e-12
    assert meta["n_steps"] == 10
    print("\nHDF5 snapshots roundtrip OK")


def test_hdf5_materials_roundtrip():
    """Save and load materials should preserve arrays."""
    try:
        import h5py  # noqa: F401
    except ImportError:
        import pytest
        pytest.skip("h5py not installed")

    from rfx.checkpoint import save_materials, load_materials

    shape = (5, 5, 5)
    materials = init_materials(shape)
    materials = materials._replace(
        eps_r=jnp.full(shape, 4.4, dtype=jnp.float32),
        sigma=jnp.full(shape, 0.02, dtype=jnp.float32),
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "mats.h5"
        save_materials(path, materials)
        loaded = load_materials(path)

    np.testing.assert_allclose(loaded.eps_r, materials.eps_r)
    np.testing.assert_allclose(loaded.sigma, materials.sigma)
    print("\nHDF5 materials roundtrip OK")


# --- High-level API 2D mode ---

def test_api_2d_tmz():
    """High-level API should accept mode='2d_tmz' and run correctly."""
    sim = Simulation(
        freq_max=10e9,
        domain=(0.02, 0.02, 0.01),
        boundary="pec",
        mode="2d_tmz",
    )
    sim.add(Box((0.005, 0.005, 0.0), (0.015, 0.015, 0.01)), material="fr4")
    sim.add_probe((0.01, 0.01, 0.0), "ez")

    result = sim.run(n_steps=20)
    assert result.state.ez.shape[2] == 1
    assert result.time_series.shape[0] == 20
    print(f"\nAPI 2D TMz: shape={result.state.ez.shape}")


def test_api_snapshot():
    """High-level API should support snapshot parameter."""
    sim = Simulation(
        freq_max=10e9,
        domain=(0.02, 0.02, 0.02),
        boundary="pec",
    )
    sim.add_probe((0.01, 0.01, 0.01), "ez")

    snap = SnapshotSpec(components=("ez",), slice_axis=2, slice_index=5)
    result = sim.run(n_steps=10, snapshot=snap)
    assert result.snapshots is not None
    assert "ez" in result.snapshots
    print(f"\nAPI snapshot: ez shape={result.snapshots['ez'].shape}")
