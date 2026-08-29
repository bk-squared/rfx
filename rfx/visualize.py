"""Field visualization utilities.

Provides matplotlib-based plotting for field snapshots, S-parameters,
and radiation patterns.  All functions return the figure for further
customization or saving.
"""

from __future__ import annotations

import warnings

import numpy as np

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def _require_mpl():
    if not HAS_MPL:
        raise ImportError("matplotlib is required for visualization")


def plot_field_slice(
    state,
    grid,
    *,
    component: str = "ez",
    axis: str = "z",
    index: int | None = None,
    title: str | None = None,
    cmap: str = "RdBu_r",
    vmax: float | None = None,
) -> object:
    """Plot a 2D slice of a field component.

    Parameters
    ----------
    state : FDTDState
    grid : Grid
    component : field name ("ex", "ey", "ez", "hx", "hy", "hz")
    axis : normal axis for the slice ("x", "y", or "z")
    index : grid index along axis (default: center)
    title : plot title
    cmap : colormap
    vmax : symmetric color range [-vmax, vmax]

    Returns
    -------
    matplotlib Figure
    """
    _require_mpl()
    field = np.asarray(getattr(state, component))

    axis_idx = {"x": 0, "y": 1, "z": 2}[axis]
    if index is None:
        index = field.shape[axis_idx] // 2

    if axis_idx == 0:
        slc = field[index, :, :]
        xlabel, ylabel = "y (cells)", "z (cells)"
    elif axis_idx == 1:
        slc = field[:, index, :]
        xlabel, ylabel = "x (cells)", "z (cells)"
    else:
        slc = field[:, :, index]
        xlabel, ylabel = "x (cells)", "y (cells)"

    if vmax is None:
        vmax = float(np.max(np.abs(slc))) or 1.0

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(slc.T, origin="lower", cmap=cmap, vmin=-vmax, vmax=vmax,
                   aspect="equal")
    fig.colorbar(im, ax=ax, label=component)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title or f"{component} slice ({axis}={index})")
    return fig


def plot_geometry_2d_slice(
    sim,
    *,
    axis: int = 2,
    index: int | None = None,
    title: str | None = None,
    cmap: str = "viridis",
    figsize: tuple[float, float] = (7, 5),
) -> object:
    """Render a 2D relative-permittivity (εr) cross-section of the geometry.

    Builds the simulation grid, assembles the material arrays, and draws a
    single slice of ``eps_r`` with a colorbar so a reader can see the dielectric
    structure (substrate, coating layers, slab) at a glance.

    Parameters
    ----------
    sim : Simulation
        rfx Simulation with geometry/materials defined.
    axis : int
        Normal axis for the slice (0=x, 1=y, 2=z). The slice plane is the
        remaining two axes.
    index : int or None
        Grid index along *axis*. ``None`` selects the centre cell.
    title : str or None
        Plot title.
    cmap : str
        Matplotlib colormap for the εr field.
    figsize : tuple
        Figure size in inches.

    Returns
    -------
    matplotlib Figure
    """
    _require_mpl()

    grid = sim._build_grid()
    eps = np.asarray(sim._assemble_materials(grid)[0].eps_r)

    if axis not in (0, 1, 2):
        raise ValueError(f"axis must be 0, 1, or 2, got {axis!r}")
    if index is None:
        # Pick the slice that actually contains the structure rather than the
        # geometric centre: for thin "1D-equivalent" domains (a layered stack
        # only a cell or two thick along y/z) the centre cell can land on a
        # padding plane that is pure vacuum. Choose the plane along *axis* with
        # the most permittivity variation; fall back to the centre if uniform.
        moved = np.moveaxis(eps, axis, 0)
        per_plane_spread = np.ptp(moved.reshape(moved.shape[0], -1), axis=1)
        if float(per_plane_spread.max()) > 0:
            index = int(np.argmax(per_plane_spread))
        else:
            index = eps.shape[axis] // 2

    if axis == 0:
        slc = eps[index, :, :]
        xlabel, ylabel = "y (mm)", "z (mm)"
    elif axis == 1:
        slc = eps[:, index, :]
        xlabel, ylabel = "x (mm)", "z (mm)"
    else:
        slc = eps[:, :, index]
        xlabel, ylabel = "x (mm)", "y (mm)"

    dx_mm = float(grid.dx) * 1e3
    extent = [0.0, slc.shape[0] * dx_mm, 0.0, slc.shape[1] * dx_mm]

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(
        slc.T,
        origin="lower",
        cmap=cmap,
        aspect="auto",
        extent=extent,
        vmin=float(slc.min()),
        vmax=float(slc.max()),
    )
    fig.colorbar(im, ax=ax, label="relative permittivity εᵣ")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title or "Geometry (εᵣ cross-section)")
    fig.tight_layout()
    return fig


def _slice_coords(sim, grid):
    """E-node coordinates for whichever lane this grid belongs to.

    Both helpers place the first INTERIOR node at 0 and give negative
    coordinates to the CPML pad, so a ``Box((0,0,0),(Lx,Ly,Lz))`` tiles the
    interior exactly and the drawn axes read in design coordinates.
    """
    from rfx.geometry.rasterize_grid import (
        coords_from_nonuniform_grid, coords_from_uniform_grid)
    from rfx.nonuniform import NonUniformGrid
    c = (coords_from_nonuniform_grid(grid) if isinstance(grid, NonUniformGrid)
         else coords_from_uniform_grid(grid))
    return [np.asarray(c.x, dtype=float), np.asarray(c.y, dtype=float),
            np.asarray(c.z, dtype=float)]


def _declared_entries(sim):
    """Every declared body, from BOTH registries.

    ``Simulation.add()`` appends to ``sim._geometry``; ``add_thin_conductor()``
    appends to ``sim._thin_conductors``. A viewer that walks only the first
    silently omits every trace on a PCB.
    """
    out = list(getattr(sim, "_geometry", []) or [])
    out += list(getattr(sim, "_thin_conductors", []) or [])
    return out


def _axis_edges(grid, axis, nodes):
    """Cell edges along *axis* in metres.

    CONVENTION, verified against the grid's own spacing arrays: an E-node
    coordinate is the LOWER EDGE of its cell, i.e. ``node[k+1] - node[k] ==
    dz[k]``. Cell k therefore spans ``[node[k], node[k] + d[k])`` and the N+1
    edges pcolormesh needs are the N nodes plus one closing edge.

    Treating the nodes as cell CENTRES (edges at their midpoints) shifts
    every drawn cell by half a cell and, on a graded axis, distorts the
    widths as well. That is the same node-vs-cell confusion that makes
    ``position=`` land one plane above a one-cell sheet, so it is pinned by
    a test rather than left to a comment.
    """
    n = np.asarray(nodes, dtype=float)
    names = ("dx_arr", "dy_arr", "dz")
    d = getattr(grid, names[axis], None)
    if d is None:
        step = float(getattr(grid, "dx"))
        d = np.full(n.size, step, dtype=float)
    else:
        d = np.asarray(d, dtype=float)
        if d.size < n.size:            # scalar-ish or short: pad with the last
            d = np.concatenate([d, np.full(n.size - d.size, d[-1])])
    if n.size == 0:
        return np.array([0.0, 1.0])
    return np.concatenate([n, [n[-1] + d[n.size - 1]]])


def plot_rasterized_slice(
    sim,
    *,
    axis: int = 2,
    position: float | None = None,
    index: int | None = None,
    show_declared: bool = True,
    sigma_threshold: float | None = None,
    figsize: tuple[float, float] = (8.0, 6.0),
    title: str | None = None,
    ax=None,
):
    """Draw the cells the SOLVE will use: conductors on top of permittivity.

    :func:`plot_geometry_2d_slice` answers "what permittivity did I ask
    for". This answers "what did the rasterizer actually build", which is a
    different question and the one that goes wrong quietly:

    * a one-cell PEC sheet carries no permittivity contrast of its own, so
      **metal is invisible in an eps_r plot** — the conductor layer here is
      drawn from :meth:`Simulation.conductor_mask`, which covers all three
      places a conductor can live (``pec_mask``, ``sigma`` above the
      conductor threshold, and the node-thin surface-impedance sheet
      operator of #677, which touches neither of the first two);
    * on a graded mesh the cells are not the same size, so the plot uses
      ``pcolormesh`` on true cell edges rather than ``imshow`` with one
      ``extent``;
    * the grid is the one this simulation would RUN on — the non-uniform
      grid whenever any of ``dx_profile`` / ``dy_profile`` / ``dz_profile``
      is set.

    Parameters
    ----------
    sim : Simulation
    axis : int
        Slice normal (0=x, 1=y, 2=z).
    position : float or None
        Physical coordinate (metres) along *axis* to slice at. A one-cell
        body is rasterized onto the node NEAREST ITS MIDPOINT
        (``rfx/geometry/csg.py`` ``_axis_mask``), and on a float32 tie that
        can round either way — so the plane holding a sheet is not reliably
        the one nearest the coordinate you asked for. The search therefore
        looks at the nearest node and its two neighbours and takes whichever
        holds conductor cells; when it moves, the title says so, because
        silently answering about a different plane turns "is this plane clear
        of metal?" into a picture of the ground plane. Raises if *position*
        is outside the axis. Mutually exclusive with ``index``.
    index : int or None
        Grid index along *axis*. Overrides ``position``. If both are None,
        the plane with the most conductor cells is used.
    show_declared : bool
        Overlay the declared outline of every entry that HAS one — i.e. an
        analytic box. Patterned bodies (imported meshes, sheets carved from
        CAD) are skipped rather than drawn as their bounding box: on a real
        board a divider arm measured 41% copper inside its own bbox, so a
        bbox rectangle reads as "this was declared solid" and is worse than
        no rectangle. The count of skipped bodies is put in the title so the
        omission is visible rather than silent.
    sigma_threshold : float or None
        Passed to :meth:`Simulation.conductor_mask`.
    figsize, title, ax
        Usual matplotlib controls. ``ax`` draws into an existing axes.

    Returns
    -------
    matplotlib Figure

    Examples
    --------
    >>> fig = plot_rasterized_slice(sim, axis=2, position=1.6e-3)  # doctest: +SKIP
    """
    _require_mpl()
    if axis not in (0, 1, 2):
        raise ValueError(f"axis must be 0, 1, or 2, got {axis!r}")
    if index is not None and position is not None:
        raise ValueError("pass index or position, not both")

    from rfx.nonuniform import NonUniformGrid
    is_nu = (sim._dx_profile is not None or sim._dy_profile is not None
             or sim._dz_profile is not None)
    grid = sim._build_nonuniform_grid() if is_nu else sim._build_grid()
    cond = np.asarray(sim.conductor_mask(grid, sigma_threshold=sigma_threshold),
                      dtype=bool)
    if isinstance(grid, NonUniformGrid):
        eps = np.asarray(sim._assemble_materials_nu(grid)[0].eps_r, dtype=float)
    else:
        eps = np.asarray(sim._assemble_materials(grid)[0].eps_r, dtype=float)
    coords = _slice_coords(sim, grid)

    names_of = lambda a: "xyz"[a]
    moved_note = ""
    n_along = cond.shape[axis]
    per_plane = np.moveaxis(cond, axis, 0).reshape(n_along, -1).sum(axis=1)
    if index is None:
        if position is None:
            index = int(np.argmax(per_plane)) if per_plane.max() else n_along // 2
        else:
            pos = float(position)
            lo_c, hi_c = float(coords[axis].min()), float(coords[axis].max())
            if not (lo_c - 1e-9 <= pos <= hi_c + 1e-9):
                raise ValueError(
                    f"position={pos:g} m is outside axis {names_of(axis)} "
                    f"[{lo_c:g}, {hi_c:g}] m. Clamping it would have drawn a "
                    "plausible empty CPML plane, which is how a metre/mm slip "
                    "reads as a clean result.")
            k = int(np.argmin(np.abs(coords[axis] - pos)))
            cand = [c for c in (k, k + 1, k - 1) if 0 <= c < n_along]
            index = max(cand, key=lambda c: int(per_plane[c]))
            if index != k:
                # The search is a convenience for finding a sheet; silently
                # answering about a DIFFERENT plane turns "is z=0.1 mm clear
                # of metal?" into a picture of the ground plane.
                moved_note = (
                    f"  [asked for {names_of(axis)} = {pos * 1e3:.4f} mm "
                    f"(plane {k}, {int(per_plane[k])} conductor cells); showing "
                    f"the neighbouring plane {index} instead because it holds "
                    f"{int(per_plane[index])}. Pass index= to override.]")
            else:
                moved_note = ""
    if not 0 <= index < n_along:
        raise IndexError(f"index {index} outside axis {axis} of length {n_along}")

    keep = [a for a in (0, 1, 2) if a != axis]
    take = [slice(None)] * 3
    take[axis] = index
    eps2, cond2 = eps[tuple(take)], cond[tuple(take)]
    xe = _axis_edges(grid, keep[0], coords[keep[0]]) * 1e3
    ye = _axis_edges(grid, keep[1], coords[keep[1]]) * 1e3

    fig = ax.figure if ax is not None else None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    # A uniform slice has a degenerate range, and matplotlib's `nonsingular`
    # then invents one: an all-vacuum slice drew a colorbar running 0.900 to
    # 1.100, i.e. eps_r < 1 on an air plane. Say the single value instead.
    lo_e, hi_e = float(eps2.min()), float(eps2.max())
    uniform = (hi_e - lo_e) <= 1e-9 * max(abs(hi_e), 1.0)
    mesh = ax.pcolormesh(xe, ye, eps2.T, cmap="Blues", shading="flat",
                         vmin=lo_e - 0.5 if uniform else lo_e,
                         vmax=lo_e + 0.5 if uniform else hi_e)
    cbar = fig.colorbar(mesh, ax=ax,
                        label="relative permittivity \u03b5\u1d63")
    if uniform:
        cbar.set_ticks([lo_e])
        cbar.set_ticklabels([f"{lo_e:.4g} (uniform)"])
    ax.pcolormesh(xe, ye, np.where(cond2, 1.0, np.nan).T, shading="flat",
                  cmap=_conductor_cmap(), vmin=0.0, vmax=1.0)

    names = "xyz"
    n_skipped = 0
    if show_declared:
        drawn = 0
        # BOTH registries. add_thin_conductor() bodies live in
        # sim._thin_conductors, not sim._geometry, so iterating only the
        # latter drew a board of red cells with no outline AND no skip note —
        # the honesty mechanism firing for Cylinder/Sphere and never for the
        # one class a PCB is actually made of.
        for entry in _declared_entries(sim):
            shape = getattr(entry, "shape", entry)
            lo = getattr(shape, "corner_lo", None)
            hi = getattr(shape, "corner_hi", None)
            if lo is None or hi is None:
                # No analytic box -> its bounds are a BOUNDING BOX, not an
                # outline. Drawing it would assert a solid rectangle the
                # model never declared.
                n_skipped += 1
                continue
            lo = [float(v) for v in lo]
            hi = [float(v) for v in hi]
            # Tolerance scaled to the CELL, not an absolute 1e-12 m: node
            # coordinates are float32 (quantum ~9e-10 m at 9 mm), so an
            # absolute 1e-12 m dropped a body's outline whenever the slice
            # sat exactly on its declared face.
            tol = 0.05 * float(np.median(np.diff(coords[axis]))) if coords[axis].size > 1 else 1e-9
            if not (lo[axis] - tol <= float(coords[axis][index]) <= hi[axis] + tol):
                continue
            ax.add_patch(plt.Rectangle(
                (lo[keep[0]] * 1e3, lo[keep[1]] * 1e3),
                (hi[keep[0]] - lo[keep[0]]) * 1e3,
                (hi[keep[1]] - lo[keep[1]]) * 1e3,
                fill=False, edgecolor="#d55e00", lw=1.4, ls="--",
                label=("declared outline (analytic box)"
                       if drawn == 0 else None), zorder=5))
            drawn += 1
        if drawn:
            ax.legend(fontsize=8, loc="upper right", framealpha=0.9)

    ax.set_xlabel(f"{names[keep[0]]} (mm)")
    ax.set_ylabel(f"{names[keep[1]]} (mm)")
    ax.set_aspect("equal", adjustable="box")
    skip_note = (f"; {n_skipped} patterned body/bodies have no analytic "
                 "outline and are not outlined" if n_skipped else "")
    ax.set_title(title or (
        f"Rasterized {names[axis]}-slice at index {index} "
        f"({names[axis]} = {float(coords[axis][index]) * 1e3:.4f} mm) — "
        f"{int(cond2.sum())} conductor cells{skip_note}{moved_note}"),
        fontsize=10)
    fig.tight_layout()
    return fig


def plot_stack_profile(
    sim,
    *,
    axis: int = 2,
    at: tuple[float, float] | None = None,
    sigma_threshold: float | None = None,
    figsize: tuple[float, float] = (7.5, 7.0),
    title: str | None = None,
    ax=None,
):
    """Layer stack along one column: declared spans beside the realized cells.

    The question this answers is "is my 6-layer board in the mesh the board I
    drew". On a layered structure the failure is not visible in a plan view:
    a laminate rounds to a different cell count, a foil lands one node high,
    a sheet's own cell keeps the permittivity of whatever abuts it. All three
    are a z-column reading.

    Left column  — every geometry entry that spans this column, drawn over its
    DECLARED extent along *axis*, labelled by material.
    Right column — the cells the solve uses, each drawn at its true size, tinted
    by assembled permittivity, with conductor cells hatched.

    Parameters
    ----------
    sim : Simulation
    axis : int
        Stack normal (0=x, 1=y, 2=z). Default 2 — the usual PCB normal.
    at : (float, float) or None
        Physical coordinates (metres) of the column in the two remaining axes.
        ``None`` picks the column carrying the most conductor cells, which is
        the one worth reading on a patterned board.
    sigma_threshold : float or None
        Passed to :meth:`Simulation.conductor_mask`.

    Returns
    -------
    matplotlib Figure
    """
    _require_mpl()
    if axis not in (0, 1, 2):
        raise ValueError(f"axis must be 0, 1, or 2, got {axis!r}")
    from rfx.nonuniform import NonUniformGrid

    is_nu = (sim._dx_profile is not None or sim._dy_profile is not None
             or sim._dz_profile is not None)
    grid = sim._build_nonuniform_grid() if is_nu else sim._build_grid()
    cond = np.asarray(sim.conductor_mask(grid, sigma_threshold=sigma_threshold),
                      dtype=bool)
    if isinstance(grid, NonUniformGrid):
        eps = np.asarray(sim._assemble_materials_nu(grid)[0].eps_r, dtype=float)
    else:
        eps = np.asarray(sim._assemble_materials(grid)[0].eps_r, dtype=float)
    coords = _slice_coords(sim, grid)
    keep = [a for a in (0, 1, 2) if a != axis]
    names_of = lambda a: "xyz"[a]
    auto_note = ""

    if at is None:
        counts = np.moveaxis(cond, axis, -1).sum(axis=-1)
        if int(counts.max()) > 0:
            i0, i1 = np.unravel_index(int(np.argmax(counts)), counts.shape)
        else:
            # No conductor anywhere — a dielectric resonator, a lens, a radome.
            # argmax on an all-zero array returns 0, i.e. the CPML CORNER, and
            # the figure comes back blank with no hint that the column was
            # chosen badly. Fall back to the column with the most permittivity
            # structure, and say so in the title.
            var = np.moveaxis(eps, axis, -1)
            # np.ptp(...), not ndarray.ptp(...) — the method was removed in
            # numpy 2.0 and this repo runs numpy>=2.
            var = np.ptp(var.reshape(var.shape[0], var.shape[1], -1), axis=-1)
            if float(var.max()) <= 0:
                raise ValueError(
                    "no conductor and no permittivity structure along "
                    f"{names_of(axis)} — there is no column worth reading, and "
                    "picking one would return a blank two-column figure that "
                    "looks like a build failure. Pass at=(u, v) if you want a "
                    "specific column anyway.")
            i0, i1 = np.unravel_index(int(np.argmax(var)), var.shape)
            auto_note = (" (no conductor in this model; column chosen by "
                         "\u03b5 structure)")
    else:
        i0 = int(np.argmin(np.abs(coords[keep[0]] - float(at[0]))))
        i1 = int(np.argmin(np.abs(coords[keep[1]] - float(at[1]))))
    take = [0, 0, 0]
    take[keep[0]], take[keep[1]], take[axis] = i0, i1, slice(None)
    eps_col = eps[tuple(take)]
    cond_col = cond[tuple(take)]
    edges = _axis_edges(grid, axis, coords[axis]) * 1e3

    fig = ax.figure if ax is not None else None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # realized cells, true sizes
    lo_e, hi_e = edges[:-1], edges[1:]
    # ABSOLUTE shading, not per-column normalisation. Renormalising made a
    # column of solid eps_r 9 pixel-identical to a column of vacuum, and the
    # docstring promises the tint carries permittivity. Anchor 1.0 to white
    # and cap at 12 so two columns (and two calls) are comparable; the tick
    # labels carry the exact values regardless.
    eps_hi = 12.0
    for k in range(eps_col.size):
        shade = 0.97 - 0.55 * min(max(float(eps_col[k]) - 1.0, 0.0),
                                  eps_hi - 1.0) / (eps_hi - 1.0)
        ax.add_patch(plt.Rectangle((1.15, lo_e[k]), 1.0, hi_e[k] - lo_e[k],
                                   facecolor=str(max(shade, 0.0)),
                                   edgecolor="0.55", lw=0.4))
        if cond_col[k]:
            ax.add_patch(plt.Rectangle((1.15, lo_e[k]), 1.0, hi_e[k] - lo_e[k],
                                       facecolor="#b03000", alpha=0.85,
                                       edgecolor="#701e00", lw=0.5))

    # declared spans of everything covering this column
    x_at = float(coords[keep[0]][i0])
    y_at = float(coords[keep[1]][i1])
    drawn = 0
    for entry in _declared_entries(sim):
        shape = getattr(entry, "shape", entry)
        lo = getattr(shape, "corner_lo", None)
        hi = getattr(shape, "corner_hi", None)
        if lo is None or hi is None:
            continue
        lo = [float(v) for v in lo]
        hi = [float(v) for v in hi]
        tolx = 0.05 * float(np.median(np.diff(coords[keep[0]]))) if coords[keep[0]].size > 1 else 1e-9
        toly = 0.05 * float(np.median(np.diff(coords[keep[1]]))) if coords[keep[1]].size > 1 else 1e-9
        if not (lo[keep[0]] - tolx <= x_at <= hi[keep[0]] + tolx
                and lo[keep[1]] - toly <= y_at <= hi[keep[1]] + toly):
            continue
        mat = (getattr(entry, "material_name", None)
               or getattr(entry, "material", None))
        ax.add_patch(plt.Rectangle((0.0, lo[axis] * 1e3), 1.0,
                                   (hi[axis] - lo[axis]) * 1e3,
                                   facecolor="#0072b2", alpha=0.30,
                                   edgecolor="#0072b2", lw=0.9))
        ax.text(0.5, 0.5 * (lo[axis] + hi[axis]) * 1e3, str(mat or "")[:14],
                ha="center", va="center", fontsize=7)
        drawn += 1

    names = "xyz"
    ax.set_xlim(-0.1, 2.35)
    ax.set_ylim(edges.min(), edges.max())
    ax.set_xticks([0.5, 1.65])
    ax.set_xticklabels(["declared", "meshed"])
    ax.set_ylabel(f"{names[axis]} (mm)")
    ax.set_title(title or (
        f"Stack along {names[axis]} at {names[keep[0]]}={x_at * 1e3:.3f} mm, "
        f"{names[keep[1]]}={y_at * 1e3:.3f} mm — {drawn} declared bod(y/ies), "
        f"{int(cond_col.sum())} conductor cell(s) (red){auto_note}"),
        fontsize=10)
    fig.tight_layout()
    return fig


def _conductor_cmap():
    from matplotlib.colors import ListedColormap
    return ListedColormap(["#b03000"])


def plot_s_params(
    s_params: np.ndarray,
    freqs: np.ndarray,
    *,
    db: bool = True,
    title: str = "S-Parameters",
) -> object:
    """Plot S-parameter magnitudes vs frequency.

    Parameters
    ----------
    s_params : (n_ports, n_ports, n_freqs) complex
    freqs : (n_freqs,) Hz
    db : plot in dB
    title : plot title

    Returns
    -------
    matplotlib Figure
    """
    _require_mpl()
    n_ports = s_params.shape[0]
    freqs_ghz = np.asarray(freqs) / 1e9

    fig, ax = plt.subplots(figsize=(8, 5))
    for i in range(n_ports):
        for j in range(n_ports):
            mag = np.abs(s_params[i, j, :])
            if db:
                y = 20 * np.log10(np.maximum(mag, 1e-10))
                ylabel = "Magnitude (dB)"
            else:
                y = mag
                ylabel = "Magnitude"
            ax.plot(freqs_ghz, y, label=f"S{i+1}{j+1}")

    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig


def plot_radiation_pattern(
    ff,
    *,
    freq_idx: int = 0,
    phi_idx: int = 0,
    db: bool = True,
    title: str | None = None,
) -> object:
    """Plot radiation pattern in polar coordinates.

    Parameters
    ----------
    ff : FarFieldResult
    freq_idx : frequency index
    phi_idx : phi cut index
    db : plot in dB (normalized)
    title : plot title

    Returns
    -------
    matplotlib Figure
    """
    _require_mpl()
    theta = ff.theta
    power = np.abs(ff.E_theta[freq_idx, :, phi_idx]) ** 2 + \
            np.abs(ff.E_phi[freq_idx, :, phi_idx]) ** 2

    peak = np.max(power)
    if peak > 0:
        power_norm = power / peak
    else:
        power_norm = power

    if db:
        r = 10 * np.log10(np.maximum(power_norm, 1e-10))
        r = np.maximum(r, -40)  # clip at -40 dB
        r = r + 40  # shift so -40 dB = 0
    else:
        r = power_norm

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"projection": "polar"})
    ax.plot(theta, r)
    ax.plot(-theta + 2 * np.pi, r)  # mirror for full pattern
    freq_ghz = ff.freqs[freq_idx] / 1e9
    ax.set_title(title or f"Radiation Pattern ({freq_ghz:.2f} GHz)")
    return fig


def plot_time_series(
    time_series: np.ndarray,
    dt: float,
    *,
    labels: list[str] | None = None,
    title: str = "Probe Time Series",
) -> object:
    """Plot probe time series.

    Parameters
    ----------
    time_series : (n_steps, n_probes) array
    dt : timestep in seconds
    labels : probe labels
    title : plot title

    Returns
    -------
    matplotlib Figure
    """
    _require_mpl()
    ts = np.asarray(time_series)
    n_steps, n_probes = ts.shape
    t_ns = np.arange(n_steps) * dt * 1e9

    fig, ax = plt.subplots(figsize=(8, 4))
    for i in range(n_probes):
        label = labels[i] if labels else f"Probe {i}"
        ax.plot(t_ns, ts[:, i], label=label)

    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("Field amplitude")
    ax.set_title(title)
    if n_probes > 1:
        ax.legend()
    ax.grid(True, alpha=0.3)
    return fig


def plot_rcs(
    rcs_result,
    *,
    freq_idx: int = 0,
    phi_idx: int = 0,
    polar: bool = True,
    title: str | None = None,
) -> object:
    """Plot RCS pattern in polar or rectangular coordinates.

    Parameters
    ----------
    rcs_result : RCSResult
        Output from ``compute_rcs()``.
    freq_idx : int
        Frequency index to plot.
    phi_idx : int
        Phi cut index to plot.
    polar : bool
        If True, plot in polar coordinates. If False, rectangular.
    title : str or None
        Plot title.

    Returns
    -------
    matplotlib Figure
    """
    _require_mpl()

    theta = rcs_result.theta
    rcs_db = rcs_result.rcs_dbsm[freq_idx, :, phi_idx]
    freq_ghz = rcs_result.freqs[freq_idx] / 1e9

    default_title = f"RCS Pattern ({freq_ghz:.2f} GHz)"
    plot_title = title or default_title

    if polar:
        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"projection": "polar"})
        # Shift RCS values for polar display (clip and shift)
        rcs_display = np.maximum(rcs_db, np.max(rcs_db) - 40)
        rcs_display = rcs_display - np.min(rcs_display)
        ax.plot(theta, rcs_display, linewidth=1.5)
        # Mirror for symmetric display
        ax.plot(-theta + 2 * np.pi, rcs_display, linewidth=1.5, alpha=0.5)
        ax.set_title(plot_title)
    else:
        fig, ax = plt.subplots(figsize=(8, 5))
        theta_deg = np.degrees(theta)
        ax.plot(theta_deg, rcs_db, linewidth=1.5)
        ax.set_xlabel("Theta (degrees)")
        ax.set_ylabel("RCS (dBsm)")
        ax.set_title(plot_title)
        ax.grid(True, alpha=0.3)

    return fig


# ===========================================================================
# 3D interactive visualisation (plotly). Issue #38.
# ===========================================================================

def _plotly():
    try:
        import plotly.graph_objects as go
        return go
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "rfx.visualize 3D functions require plotly. "
            "Install with `pip install plotly kaleido`."
        ) from e


_MATERIAL_COLORS = {
    "pec": ("black", 0.7),
    "fr4": ("tan", 0.2),
    "glass": ("lightblue", 0.25),
}


def _material_style(mat_name):
    return _MATERIAL_COLORS.get((mat_name or "").lower(),
                                ("cornflowerblue", 0.25))


def _cuboid_trace(go, *, x0, y0, z0, w, d, h, color, opacity, name,
                  visible=True):
    """Axis-aligned box as Mesh3d. Coords in metres, rendered in mm."""
    mm = 1e3
    xs = [x0, x0 + w, x0 + w, x0,     x0,     x0 + w, x0 + w, x0]
    ys = [y0, y0,     y0 + d, y0 + d, y0,     y0,     y0 + d, y0 + d]
    zs = [z0, z0,     z0,     z0,     z0 + h, z0 + h, z0 + h, z0 + h]
    i = [0, 0, 1, 1, 2, 2, 4, 4, 0, 0, 1, 2]
    j = [1, 2, 2, 5, 3, 6, 5, 6, 4, 5, 5, 3]
    k = [2, 3, 5, 6, 6, 7, 6, 7, 5, 1, 6, 7]
    return go.Mesh3d(
        x=[v * mm for v in xs], y=[v * mm for v in ys], z=[v * mm for v in zs],
        i=i, j=j, k=k, color=color, opacity=opacity, name=name,
        flatshading=True, showlegend=True,
        visible=True if visible is True else "legendonly",
    )


def _wireframe_box(go, *, corner_lo, corner_hi, color, name,
                   dash="solid", width=2, visible="legendonly"):
    mm = 1e3
    x0, y0, z0 = [c * mm for c in corner_lo]
    x1, y1, z1 = [c * mm for c in corner_hi]
    corners = [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
               (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]
    edges = [(0, 1), (1, 2), (2, 3), (3, 0),
             (4, 5), (5, 6), (6, 7), (7, 4),
             (0, 4), (1, 5), (2, 6), (3, 7)]
    xs, ys, zs = [], [], []
    for a, b in edges:
        xs += [corners[a][0], corners[b][0], None]
        ys += [corners[a][1], corners[b][1], None]
        zs += [corners[a][2], corners[b][2], None]
    return go.Scatter3d(
        x=xs, y=ys, z=zs, mode="lines", name=name,
        line=dict(color=color, width=width, dash=dash), visible=visible,
        showlegend=True,
    )


def visualize_structure(sim, *, include_cpml: bool = True,
                        include_ntff: bool = True,
                        include_sources: bool = True,
                        title: str | None = None):
    """Render a 3D plotly scene of a Simulation's geometry.

    Each material box, source marker, NTFF wireframe, and the CPML
    outer wireframe is a separate legend-toggleable trace — click a
    legend entry to show/hide that group.
    """
    go = _plotly()
    fig = go.Figure()

    for entry in sim._geometry:
        try:
            c1, c2 = entry.shape.bounding_box()
        except Exception:
            continue
        w, d, h = [c2[i] - c1[i] for i in range(3)]
        color, opacity = _material_style(entry.material_name)
        fig.add_trace(_cuboid_trace(
            go, x0=c1[0], y0=c1[1], z0=c1[2], w=w, d=d, h=h,
            color=color, opacity=opacity, name=entry.material_name,
        ))

    if include_sources and getattr(sim, "_ports", None):
        xs, ys, zs = [], [], []
        for pe in sim._ports:
            xs.append(pe.position[0] * 1e3)
            ys.append(pe.position[1] * 1e3)
            zs.append(pe.position[2] * 1e3)
        if xs:
            fig.add_trace(go.Scatter3d(
                x=xs, y=ys, z=zs, mode="markers",
                marker=dict(size=5, color="green", symbol="diamond"),
                name="sources / ports"))

    if include_ntff and getattr(sim, "_ntff", None) is not None:
        ntff_lo, ntff_hi, _freqs = sim._ntff
        fig.add_trace(_wireframe_box(
            go, corner_lo=ntff_lo, corner_hi=ntff_hi,
            color="orange", name="NTFF box"))

    if include_cpml:
        dom_x, dom_y = sim._domain[0], sim._domain[1]
        dom_z = sim._domain[2] if len(sim._domain) > 2 else 0
        if dom_z == 0 and sim._dz_profile is not None:
            dom_z = float(np.sum(sim._dz_profile))
        elif dom_z == 0:
            dom_z = dom_x
        fig.add_trace(_wireframe_box(
            go, corner_lo=(0, 0, 0), corner_hi=(dom_x, dom_y, dom_z),
            color="purple", dash="dash", name="domain / CPML"))

    fig.update_layout(
        title=title or "rfx Simulation structure",
        legend=dict(x=0.01, y=0.99),
        scene=dict(
            xaxis=dict(title="x (mm)"), yaxis=dict(title="y (mm)"),
            zaxis=dict(title="z (mm)"), aspectmode="data",
        ),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


def visualize_farfield_3d(result, sim=None, *, f_idx: int = 0,
                          theta_grid=None, phi_grid=None,
                          geometry: bool = True,
                          opacity: float = 0.75,
                          scale_mm: float = 40.0,
                          centre=None,
                          title: str | None = None):
    """Interactive 3D far-field lobe + optional structure overlay.

    Requires ``result.ntff_data`` and ``result.ntff_box`` (add an NTFF
    box to the Simulation first). Legend-toggleable overlays for
    geometry and CPML / NTFF boundaries.
    """
    go = _plotly()
    if result.ntff_data is None or result.ntff_box is None:
        raise ValueError(
            "visualize_farfield_3d requires ntff_data/ntff_box on the "
            "result — add sim.add_ntff_box(...) and re-run."
        )
    from rfx.farfield import compute_far_field

    if theta_grid is None:
        theta_grid = np.linspace(0.01, np.pi / 2, 60)
    if phi_grid is None:
        phi_grid = np.linspace(0, 2 * np.pi, 121)

    if sim is not None:
        grid = (sim._build_nonuniform_grid()
                if sim._dz_profile is not None else sim._build_grid())
    else:
        grid = result.grid

    ef = compute_far_field(result.ntff_data, result.ntff_box, grid,
                           theta_grid, phi_grid)
    E_t = np.asarray(ef.E_theta[f_idx])
    E_p = np.asarray(ef.E_phi[f_idx])
    mag = np.sqrt(np.abs(E_t) ** 2 + np.abs(E_p) ** 2)
    mag_n = mag / np.max(mag)
    mag_db = 20 * np.log10(np.maximum(mag_n, 1e-3))

    TH, PH = np.meshgrid(theta_grid, phi_grid, indexing="ij")
    if centre is None:
        centre = (
            0.5 * sim._domain[0] * 1e3 if sim is not None else 0.0,
            0.5 * sim._domain[1] * 1e3 if sim is not None else 0.0,
            0.0,
        )
    cx, cy, cz = centre
    r = mag_n
    X = cx + scale_mm * r * np.sin(TH) * np.cos(PH)
    Y = cy + scale_mm * r * np.sin(TH) * np.sin(PH)
    Z = cz + scale_mm * r * np.cos(TH)

    freqs = np.asarray(result.ntff_box.freqs)
    f_label = f"{freqs[f_idx]/1e9:.3f} GHz"

    fig = go.Figure()
    if sim is not None and geometry:
        for tr in visualize_structure(sim, title=None).data:
            fig.add_trace(tr)
    fig.add_trace(go.Surface(
        x=X, y=Y, z=Z, surfacecolor=mag_db,
        colorscale="Viridis", cmin=-30, cmax=0, opacity=opacity,
        colorbar=dict(title="|E_far| (dB, norm)", x=1.05),
        name=f"|E_far| @ {f_label}", showlegend=True,
    ))

    i_p, j_p = np.unravel_index(np.argmax(mag), mag.shape)
    th_pk, ph_pk = theta_grid[i_p], phi_grid[j_p]
    fig.add_trace(go.Scatter3d(
        x=[cx, cx + scale_mm * 1.1 * np.sin(th_pk) * np.cos(ph_pk)],
        y=[cy, cy + scale_mm * 1.1 * np.sin(th_pk) * np.sin(ph_pk)],
        z=[cz, cz + scale_mm * 1.1 * np.cos(th_pk)],
        mode="lines+markers", line=dict(color="red", width=4),
        marker=dict(size=[3, 6], color="red"),
        name=f"peak θ={np.degrees(th_pk):.0f}° φ={np.degrees(ph_pk):.0f}°",
    ))

    fig.update_layout(
        title=title or f"Far-field lobe @ {f_label}",
        legend=dict(x=0.01, y=0.99),
        scene=dict(
            xaxis=dict(title="x (mm)"), yaxis=dict(title="y (mm)"),
            zaxis=dict(title="z (mm)"), aspectmode="data",
        ),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


def save_html(fig, path: str, include_plotlyjs: str = "cdn") -> None:
    """Write a plotly Figure to an interactive HTML file."""
    fig.write_html(path, include_plotlyjs=include_plotlyjs)


def save_png(fig, path: str, **kwargs) -> None:
    """Write a plotly Figure to PNG (requires kaleido)."""
    fig.write_image(path, **kwargs)
