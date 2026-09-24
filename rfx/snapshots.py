"""Where and when each sample of a recorded field sits (#1258, #1259).

A field array returned by a uniform-grid run -- ``Result.snapshots[comp]`` or
one component of ``Result.state`` -- is indexed on the PADDED Yee lattice: the
CPML cells are included, in the same index space as ``grid.shape``,
``grid.position_to_index`` and ``grid.node_of``. This module turns those
indices into positions in metres and frame numbers into times in seconds, so a
reader never has to re-derive rfx's lattice layout.

Positions
    Coordinates are in the model frame: ``0`` is the lower corner of the
    domain declared in ``Simulation(domain=...)`` (the first interior E node),
    so CPML samples have negative coordinates on the low side and coordinates
    beyond the domain on the high side. E node ``i`` sits at
    ``x_i = (i - pad_lo) * dx`` (``rfx.geometry.rasterize_grid``'s uniform node
    formula, the same one ``Grid.node_of`` evaluates). Each component sits half
    a cell above the node along the axes listed in its stagger:

    ======  ===================================  ================
    comp    sample of ``F[i, j, k]``             stagger (cells)
    ======  ===================================  ================
    ``ex``  ``(x_{i+1/2}, y_j, z_k)``            ``(1/2, 0, 0)``
    ``ey``  ``(x_i, y_{j+1/2}, z_k)``            ``(0, 1/2, 0)``
    ``ez``  ``(x_i, y_j, z_{k+1/2})``            ``(0, 0, 1/2)``
    ``hx``  ``(x_i, y_{j+1/2}, z_{k+1/2})``      ``(0, 1/2, 1/2)``
    ``hy``  ``(x_{i+1/2}, y_j, z_{k+1/2})``      ``(1/2, 0, 1/2)``
    ``hz``  ``(x_{i+1/2}, y_{j+1/2}, z_k)``      ``(1/2, 1/2, 0)``
    ======  ===================================  ================

    The E rows are the lattice ownership contract's index convention
    (``rfx/boundaries/pec.py``, #931 section 1.1). The H rows follow from
    ``rfx.core.yee.update_h``: ``H`` is built from FORWARD differences of E
    (``Hx`` from ``Ez[j+1] - Ez[j]`` and ``Ey[k+1] - Ey[k]``), so each H sample
    sits half a cell above the E samples it differences. The last sample along
    a staggered axis (index ``n - 1``) lies half a cell beyond the last node;
    the solver carries it but it is outside the node span.

Times
    One scan step updates H first and E second (``make_core_step`` in
    ``rfx/simulation.py``), starting from zero fields at ``t = 0``. After ``s``
    completed steps E holds ``E(s * dt)`` and H holds ``H((s - 1/2) * dt)`` --
    the same convention rfx's own flux DFT uses (E at ``step * dt``, H at
    ``step * dt - dt/2``). ``Result.state`` is the state after
    ``int(state.step)`` steps; a snapshot frame is the state after the step
    count in :attr:`SnapshotAxes.steps`.

Uniform grids only: the non-uniform, distributed and subgridded lanes do not
record snapshots (they warn and return ``snapshots=None``), and ADI refuses
them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

_COMPONENTS = ("ex", "ey", "ez", "hx", "hy", "hz")
_AXIS_NAMES = ("x", "y", "z")

#: Half-cell offset, in cells, of each component's sample from the E node with
#: the same indices. See the module docstring for where each row comes from.
_YEE_STAGGER = {
    "ex": (0.5, 0.0, 0.0),
    "ey": (0.0, 0.5, 0.0),
    "ez": (0.0, 0.0, 0.5),
    "hx": (0.0, 0.5, 0.5),
    "hy": (0.5, 0.0, 0.5),
    "hz": (0.5, 0.5, 0.0),
}


@dataclass(frozen=True, eq=False)
class SnapshotAxes:
    """Positions and times of the samples in one recorded field array.

    One per component, in ``Result.snapshot_axes[comp]`` or from
    :func:`snapshot_axes`. The array it describes has dimensions ``dims``:
    ``"frame"`` first, then the spatial axes kept by the recording, in
    ``x, y, z`` order. Every array here is read-only float64 (``steps`` is
    int64).

    Attributes
    ----------
    component : str
        ``"ex"`` ... ``"hz"``.
    dims : tuple of str
        Axis names of the recorded array, e.g. ``("frame", "x", "y")`` for a
        slice normal to z, ``("frame", "x", "y", "z")`` for a full field.
    coords : dict[str, ndarray]
        For each spatial axis in ``dims``, the coordinate in metres of every
        sample along that axis, in the model frame (see the module docstring).
        ``len(coords[a])`` equals the array's extent along ``a``; it includes
        the CPML cells.
    stagger : (float, float, float)
        This component's offset from the E node, in cells, per axis.
    slice_axis : str or None
        ``"x"``, ``"y"`` or ``"z"`` for a slice, ``None`` for the full field.
    slice_index : int or None
        The recorded ``SnapshotSpec.slice_index``, a PADDED lattice index (the
        same index space as ``grid.shape`` and ``grid.position_to_index``,
        CPML cells counted), normalised to be non-negative.
    slice_coord : float or None
        Position in metres of the plane this component's slice samples lie
        on. It includes the component's own stagger along the slice axis, so
        one ``slice_index`` gives different planes for different components
        (``ez`` at ``k`` lies on ``z_{k+1/2}``, ``ex`` at ``k`` on ``z_k``).
    interval : int
        ``SnapshotSpec.interval`` of the run.
    n_steps : int
        Time steps the run completed.
    steps : ndarray of int64
        Steps completed when each frame was taken: ``interval, 2*interval,
        ..., (n_steps // interval) * interval``.
    times_s : ndarray of float64
        Time in seconds of each frame: ``steps * dt`` for E components,
        ``(steps - 1/2) * dt`` for H components.
    dt : float
        The run's time step in seconds.
    """

    component: str
    dims: tuple
    coords: Mapping[str, np.ndarray]
    stagger: tuple
    slice_axis: str | None
    slice_index: int | None
    slice_coord: float | None
    interval: int
    n_steps: int
    steps: np.ndarray
    times_s: np.ndarray
    dt: float


def _readonly(arr: np.ndarray) -> np.ndarray:
    arr.flags.writeable = False
    return arr


def _check_component(component) -> str:
    if component not in _COMPONENTS:
        raise ValueError(
            f"field component must be one of {_COMPONENTS}, got {component!r}")
    return component


def _check_uniform_grid(grid) -> None:
    from rfx.grid import Grid

    if not isinstance(grid, Grid):
        raise NotImplementedError(
            f"snapshot sample positions are defined for a uniform rfx.Grid "
            f"only, got {type(grid).__name__}. The non-uniform, distributed "
            f"and subgridded lanes do not record field snapshots.")


def validate_snapshot_spec(snapshot) -> int:
    """Refuse a malformed ``SnapshotSpec`` and return its interval.

    ``interval`` must be an integer >= 1 (``bool`` refused), every component
    one of ``ex, ey, ez, hx, hy, hz``, ``slice_axis`` one of ``None, 0, 1, 2``
    and ``slice_index`` ``None`` or an integer.
    """
    interval = snapshot.interval
    if (isinstance(interval, bool)
            or not isinstance(interval, (int, np.integer))
            or int(interval) < 1):
        raise ValueError(
            f"SnapshotSpec.interval must be an integer >= 1 (record a frame "
            f"every `interval` time steps), got {interval!r}")
    comps = tuple(snapshot.components)
    if not comps:
        raise ValueError("SnapshotSpec.components is empty")
    for comp in comps:
        _check_component(comp)
    axis = snapshot.slice_axis
    if axis is not None and (isinstance(axis, bool) or axis not in (0, 1, 2)):
        raise ValueError(
            f"SnapshotSpec.slice_axis must be None, 0, 1 or 2, got {axis!r}")
    idx = snapshot.slice_index
    if idx is not None and (isinstance(idx, bool)
                            or not isinstance(idx, (int, np.integer))):
        raise ValueError(
            f"SnapshotSpec.slice_index must be None or an integer padded "
            f"lattice index, got {idx!r}")
    return int(interval)


def snapshot_frame_steps(n_steps: int, interval: int) -> np.ndarray:
    """Steps completed at each recorded frame: ``interval, 2*interval, ...``.

    ``n_steps // interval`` frames; a run whose length is not a multiple of
    ``interval`` records no frame for its last ``n_steps % interval`` steps.
    """
    n_steps, interval = int(n_steps), int(interval)
    if interval < 1:
        raise ValueError(f"interval must be >= 1, got {interval}")
    return np.arange(interval, n_steps + 1, interval, dtype=np.int64)


def field_sample_coords(grid, component: str) -> dict[str, np.ndarray]:
    """Coordinates in metres of every sample of one full field array.

    For the ``(nx, ny, nz)`` arrays of ``Result.state`` and of a full-field
    snapshot on a uniform ``grid``: ``{"x": (nx,), "y": (ny,), "z": (nz,)}``,
    float64, in the model frame, CPML cells included (see the module
    docstring for the layout). The field ``F[i, j, k]`` of ``component`` sits
    at ``(x[i], y[j], z[k])``.

    In 2-D mode the z axis holds one sample and its coordinate is nominal.
    """
    from rfx.geometry.rasterize_grid import coords_from_uniform_grid

    comp = _check_component(component)
    _check_uniform_grid(grid)
    nodes = coords_from_uniform_grid(grid)
    out = {}
    for ax, name in enumerate(_AXIS_NAMES):
        node = np.array((nodes.x, nodes.y, nodes.z)[ax], dtype=np.float64)
        # Sample = node + half of the primal cell it opens, derived FROM the
        # node (the direction ``geometry.smoothing._uniform_sample_coords``
        # uses). ``grid.cells`` is ``dx`` everywhere on a uniform grid.
        out[name] = _readonly(node + grid.cells(ax) * 0.5
                              if _YEE_STAGGER[comp][ax] else node)
    return out


def snapshot_axes(grid, snapshot, n_steps: int, *,
                  dt: float | None = None) -> dict[str, SnapshotAxes]:
    """Positions and times of the frames a run with ``snapshot`` records.

    Needs no run: ``grid`` is the uniform grid the run steps on
    (``sim._build_grid()``, or ``result.grid`` afterwards), ``snapshot`` the
    ``SnapshotSpec`` and ``n_steps`` the number of time steps. ``dt`` is the
    run's time step; it defaults to ``grid.dt``, which is the step of every
    uniform run except ``stencil_order=4`` (whose step is derated, and whose
    ``Result.snapshot_axes`` carries the real one). Returns
    ``{component: SnapshotAxes}``; a run returns the same thing in
    ``Result.snapshot_axes``.

    A ``SnapshotSpec`` with ``slice_axis`` set and ``slice_index=None``
    records the full field, as it always has; its axes say so
    (``slice_axis is None``).
    """
    interval = validate_snapshot_spec(snapshot)
    _check_uniform_grid(grid)
    n_steps = int(n_steps)
    if n_steps < 0:
        raise ValueError(f"n_steps must be >= 0, got {n_steps}")
    dt = float(grid.dt if dt is None else dt)
    steps = snapshot_frame_steps(n_steps, interval)
    sliced = snapshot.slice_axis is not None and snapshot.slice_index is not None
    axis = int(snapshot.slice_axis) if sliced else None
    index = None
    if sliced:
        n = int(grid.shape[axis])
        index = int(snapshot.slice_index)
        if not -n <= index < n:
            raise ValueError(
                f"SnapshotSpec.slice_index={index} is outside axis "
                f"{_AXIS_NAMES[axis]!r}, which has {n} padded samples")
        index %= n
    out = {}
    for comp in snapshot.components:
        full = field_sample_coords(grid, comp)
        if sliced:
            name = _AXIS_NAMES[axis]
            kept = tuple(a for a in _AXIS_NAMES if a != name)
            slice_coord = float(full[name][index])
        else:
            name = None
            kept = _AXIS_NAMES
            slice_coord = None
        half = 0.5 if comp.startswith("h") else 0.0
        times = (steps.astype(np.float64) - half) * dt
        out[comp] = SnapshotAxes(
            component=comp,
            dims=("frame",) + kept,
            coords={a: full[a] for a in kept},
            stagger=_YEE_STAGGER[comp],
            slice_axis=name,
            slice_index=index,
            slice_coord=slice_coord,
            interval=interval,
            n_steps=n_steps,
            steps=_readonly(steps.copy()),
            times_s=_readonly(times),
            dt=dt,
        )
    return out


def snapshot_extractor(snapshot):
    """The slicing a run applies to its state to record one frame.

    Returns ``take(state) -> list`` of arrays, one per component, in
    ``snapshot.components`` order. The full field when ``slice_axis`` or
    ``slice_index`` is ``None``.
    """
    comps = tuple(snapshot.components)
    axis = snapshot.slice_axis
    index = snapshot.slice_index

    def take(st):
        snaps = []
        for comp in comps:
            field = getattr(st, comp)
            if axis is not None and index is not None:
                sl = [slice(None)] * 3
                sl[axis] = index
                field = field[tuple(sl)]
            snaps.append(field)
        return snaps

    return take
