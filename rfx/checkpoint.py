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
    axes: dict | None = None,
) -> None:
    """Save field snapshots to an HDF5 file.

    Parameters
    ----------
    path : str or Path
    snapshots : dict mapping component name to (n_frames, ...) array
    grid : Grid or None
    dt : float or None
        Timestep for time-axis metadata. Also writes the attribute
        ``n_steps``, which (its historical name notwithstanding) is the
        number of FRAMES in the first array, and ``n_frames``, the same
        number under its right name.
    axes : dict[str, SnapshotAxes] or None
        ``Result.snapshot_axes`` (#1259). Stored per component under
        ``axes/<component>`` -- the sample coordinates, the slice plane and
        the step count and time of every frame -- and returned by
        :func:`load_snapshots` as ``metadata["axes"]``.
    """
    _require_h5py()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(path, "w") as f:
        grp = f.create_group("snapshots")
        for name, arr in snapshots.items():
            grp.create_dataset(name, data=np.array(arr))

        if dt is not None:
            n_frames = next(iter(snapshots.values())).shape[0]
            grp.attrs["dt"] = dt
            grp.attrs["n_steps"] = n_frames
            grp.attrs["n_frames"] = n_frames

        if axes is not None:
            agrp = f.create_group("axes")
            for comp, ax in axes.items():
                _write_snapshot_axes(agrp.create_group(comp), ax)

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
        File attributes; ``metadata["axes"]`` holds the
        ``{component: SnapshotAxes}`` saved with ``axes=``, when there are
        any.
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

        if "axes" in f:
            metadata["axes"] = {
                comp: _read_snapshot_axes(f["axes"][comp])
                for comp in f["axes"]}

    return snapshots, metadata


def _write_snapshot_axes(grp, ax) -> None:
    """One ``SnapshotAxes`` into an HDF5 group (``None`` as a sentinel)."""
    grp.attrs["component"] = ax.component
    grp.attrs["dims"] = [str(d) for d in ax.dims]
    grp.attrs["stagger"] = np.asarray(ax.stagger, dtype=np.float64)
    grp.attrs["slice_axis"] = "" if ax.slice_axis is None else ax.slice_axis
    grp.attrs["slice_index"] = -1 if ax.slice_index is None else int(ax.slice_index)
    grp.attrs["slice_coord"] = (np.nan if ax.slice_coord is None
                                else float(ax.slice_coord))
    grp.attrs["interval"] = int(ax.interval)
    grp.attrs["n_steps"] = int(ax.n_steps)
    grp.attrs["dt"] = float(ax.dt)
    grp.create_dataset("steps", data=np.asarray(ax.steps, dtype=np.int64))
    grp.create_dataset("times_s", data=np.asarray(ax.times_s, dtype=np.float64))
    cgrp = grp.create_group("coords")
    for name, arr in ax.coords.items():
        cgrp.create_dataset(name, data=np.asarray(arr, dtype=np.float64))


def _read_snapshot_axes(grp):
    from rfx.snapshots import SnapshotAxes

    def _ro(arr):
        arr.flags.writeable = False
        return arr

    dims = tuple(str(d) for d in grp.attrs["dims"])
    slice_axis = str(grp.attrs["slice_axis"]) or None
    slice_index = int(grp.attrs["slice_index"])
    slice_coord = float(grp.attrs["slice_coord"])
    return SnapshotAxes(
        component=str(grp.attrs["component"]),
        dims=dims,
        coords={name: _ro(grp["coords"][name][:]) for name in dims[1:]},
        stagger=tuple(float(v) for v in grp.attrs["stagger"]),
        slice_axis=slice_axis,
        slice_index=None if slice_axis is None else slice_index,
        slice_coord=None if slice_axis is None else slice_coord,
        interval=int(grp.attrs["interval"]),
        n_steps=int(grp.attrs["n_steps"]),
        steps=_ro(grp["steps"][:]),
        times_s=_ro(grp["times_s"][:]),
        dt=float(grp.attrs["dt"]),
    )


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
