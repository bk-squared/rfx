"""HDF5 checkpoint save/load for FDTD state and simulation results.

Supports saving and restoring:
- Full FDTD field state (ex, ey, ez, hx, hy, hz)
- Grid metadata (shape, dx, dt, freq_max)
- Field snapshots from mid-simulation recording
- Material arrays
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np

try:
    import h5py
except ImportError:
    h5py = None

from rfx.core.yee import FDTDState, MaterialArrays


def _require_h5py():
    if h5py is None:
        raise ImportError(
            "h5py is required for checkpoint save/load. "
            "Install with: pip install h5py"
        )


def save_state(path: str | Path, state: FDTDState, grid=None) -> None:
    """Save FDTD state to an HDF5 file.

    Parameters
    ----------
    path : str or Path
        Output file path (.h5).
    state : FDTDState
        Field state to save.
    grid : Grid or None
        If provided, grid metadata is stored as attributes.
    """
    _require_h5py()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(path, "w") as f:
        grp = f.create_group("state")
        for name in ("ex", "ey", "ez", "hx", "hy", "hz"):
            grp.create_dataset(name, data=np.array(getattr(state, name)))
        grp.attrs["step"] = int(state.step)

        if grid is not None:
            g = f.create_group("grid")
            g.attrs["shape"] = list(grid.shape)
            g.attrs["dx"] = grid.dx
            g.attrs["dt"] = grid.dt
            g.attrs["freq_max"] = grid.freq_max
            g.attrs["mode"] = grid.mode
            g.attrs["cpml_layers"] = grid.cpml_layers


def load_state(path: str | Path) -> tuple[FDTDState, dict]:
    """Load FDTD state from an HDF5 file.

    Returns
    -------
    state : FDTDState
    metadata : dict
        Grid metadata if it was saved, otherwise empty dict.
    """
    _require_h5py()
    path = Path(path)

    with h5py.File(path, "r") as f:
        grp = f["state"]
        fields = {name: jnp.array(grp[name][:]) for name in
                  ("ex", "ey", "ez", "hx", "hy", "hz")}
        step = int(grp.attrs.get("step", 0))

        state = FDTDState(
            **fields,
            step=jnp.array(step, dtype=jnp.int32),
        )

        metadata = {}
        if "grid" in f:
            g = f["grid"]
            metadata = {k: g.attrs[k] for k in g.attrs}
            if "shape" in metadata:
                metadata["shape"] = tuple(metadata["shape"])

    return state, metadata


def save_snapshots(
    path: str | Path,
    snapshots: dict[str, jnp.ndarray],
    grid=None,
    dt: float | None = None,
) -> None:
    """Save field snapshots to an HDF5 file.

    Parameters
    ----------
    path : str or Path
    snapshots : dict mapping component name to (n_steps, ...) array
    grid : Grid or None
    dt : float or None
        Timestep for time-axis metadata.
    """
    _require_h5py()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(path, "w") as f:
        grp = f.create_group("snapshots")
        for name, arr in snapshots.items():
            grp.create_dataset(name, data=np.array(arr))

        if dt is not None:
            n_steps = next(iter(snapshots.values())).shape[0]
            grp.attrs["dt"] = dt
            grp.attrs["n_steps"] = n_steps

        if grid is not None:
            g = f.create_group("grid")
            g.attrs["shape"] = list(grid.shape)
            g.attrs["dx"] = grid.dx
            g.attrs["dt"] = grid.dt
            g.attrs["freq_max"] = grid.freq_max
            g.attrs["mode"] = grid.mode


def load_snapshots(path: str | Path) -> tuple[dict[str, np.ndarray], dict]:
    """Load field snapshots from an HDF5 file.

    Returns
    -------
    snapshots : dict mapping component name to numpy array
    metadata : dict
    """
    _require_h5py()
    path = Path(path)

    with h5py.File(path, "r") as f:
        grp = f["snapshots"]
        snapshots = {name: grp[name][:] for name in grp}
        metadata = dict(grp.attrs)

        if "grid" in f:
            g = f["grid"]
            metadata.update({k: g.attrs[k] for k in g.attrs})
            if "shape" in metadata:
                metadata["shape"] = tuple(metadata["shape"])

    return snapshots, metadata


def save_materials(path: str | Path, materials: MaterialArrays) -> None:
    """Save material arrays to an HDF5 file."""
    _require_h5py()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(path, "w") as f:
        grp = f.create_group("materials")
        grp.create_dataset("eps_r", data=np.array(materials.eps_r))
        grp.create_dataset("sigma", data=np.array(materials.sigma))
        grp.create_dataset("mu_r", data=np.array(materials.mu_r))


def load_materials(path: str | Path) -> MaterialArrays:
    """Load material arrays from an HDF5 file."""
    _require_h5py()
    path = Path(path)

    with h5py.File(path, "r") as f:
        grp = f["materials"]
        return MaterialArrays(
            eps_r=jnp.array(grp["eps_r"][:]),
            sigma=jnp.array(grp["sigma"][:]),
            mu_r=jnp.array(grp["mu_r"][:]),
        )
