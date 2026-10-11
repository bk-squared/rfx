"""Read-only, cell-volume snapshots of the shared production realization."""

from dataclasses import dataclass
from typing import Literal

import numpy as np

from rfx.grid import C0
from rfx.model.materials import volume_cells


@dataclass(frozen=True, kw_only=True)
class RealizedView:
    """A NumPy snapshot, with no solver or component-material operands.

    ``nodes[a]`` has length ``shape[a]``: the context's low-corner node
    for each stored cell index, in metres from the declared domain origin.
    Absorber nodes are included. Nonperiodic axes also store a high bounding
    node; ``interior`` contains half-open physical CELL ranges, excluding
    that bounding entry and absorber pads (a singleton axis is retained).
    ``eps_r`` and ``sigma`` are float64 volume arrays shaped ``shape``, with
    lumped stamps removed; wire operands are not included. All arrays are
    independent, non-writeable copies of the shared realization.

    ``min_cells_per_wavelength`` is the interior minimum of
    ``C0 / (freq_max * sqrt(eps_r * mu_r) * max(dx, dy, dz))``, using each
    cell's volume materials and edge lengths. ``dt`` is in seconds.
    """

    nodes: tuple[np.ndarray, np.ndarray, np.ndarray]
    interior: tuple[tuple[int, int], tuple[int, int], tuple[int, int]]
    eps_r: np.ndarray
    sigma: np.ndarray
    shape: tuple[int, int, int]
    dt: float
    lane: Literal["uniform", "nonuniform"]
    min_cells_per_wavelength: float


def _snapshot(array):
    result = np.array(array, dtype=np.float64, copy=True)
    result.setflags(write=False)
    return result


def realized_view(sim) -> RealizedView:
    """Read the campaign context cached by preflight for this configuration.

    Builds the shared assembly lazily when needed, without solving or
    stepping. A traced mesh or a failed realization raises ``ValueError``
    with the context's error text; no partial snapshot is returned.
    """
    ctx = sim._campaign_ctx()
    if ctx.error is not None:
        raise ValueError(ctx.error)
    assembled = ctx.assembled()
    if assembled is None:
        raise ValueError(ctx.assembly_error)
    materials, _ = assembled
    grid = ctx.grid
    shape = tuple(int(n) for n in grid.shape)
    interior = tuple(
        (int(getattr(grid, f"pad_{axis}_lo")),
         n - int(getattr(grid, f"pad_{axis}_hi"))
         - int(n > 1 and not ctx.periodic[a]))
        for a, (axis, n) in enumerate(zip("xyz", shape))
    )
    region = tuple(slice(lo, hi) for lo, hi in interior)
    eps_volume, sigma = volume_cells(materials)
    eps_r = _snapshot(eps_volume)
    edges = np.meshgrid(*(np.asarray(d, dtype=np.float64)[s]
                          for d, s in zip(ctx.cell_sizes, region)), indexing="ij")
    h_max = np.maximum(np.maximum(edges[0], edges[1]), edges[2])
    mu_r = np.asarray(materials.mu_r, dtype=np.float64)[region]
    cells_per_wavelength = C0 / (
        float(sim._freq_max) * np.sqrt(eps_r[region] * mu_r) * h_max)
    return RealizedView(
        nodes=tuple(_snapshot(nodes) for nodes in ctx.nodes),
        interior=interior, eps_r=eps_r, sigma=_snapshot(sigma),
        shape=shape, dt=float(grid.dt), lane=ctx.lane,
        min_cells_per_wavelength=float(np.min(cells_per_wavelength)),
    )
