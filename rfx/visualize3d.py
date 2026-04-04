"""3D visualization for rfx FDTD simulations.

Provides geometry rendering, field visualization, and VTK export.
Uses matplotlib 3D as default backend, with optional pyvista for
interactive/publication-quality rendering.
"""

from __future__ import annotations

import numpy as np

try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: F401
    HAS_MPL3D = True
except (ImportError, Exception):
    # mpl_toolkits.mplot3d may fail with dual matplotlib installations
    HAS_MPL3D = False
    try:
        import matplotlib.pyplot as plt
        HAS_MPL = True
    except ImportError:
        HAS_MPL = False

try:
    import pyvista as pv
    HAS_PYVISTA = True
except ImportError:
    HAS_PYVISTA = False


def _has_mpl():
    """Check if any form of matplotlib is available."""
    return HAS_MPL3D or globals().get("HAS_MPL", False)


def _require_mpl():
    if not _has_mpl():
        raise ImportError("matplotlib is required for visualization")


# ---------------------------------------------------------------------------
# Geometry rendering
# ---------------------------------------------------------------------------

def _box_faces(p0, p1, color="C0", alpha=0.3):
    """Generate 6 face polygons for a box defined by corners p0, p1."""
    x0, y0, z0 = p0
    x1, y1, z1 = p1
    verts = [
        [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0)],  # bottom
        [(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)],  # top
        [(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)],  # front
        [(x0, y1, z0), (x1, y1, z0), (x1, y1, z1), (x0, y1, z1)],  # back
        [(x0, y0, z0), (x0, y1, z0), (x0, y1, z1), (x0, y0, z1)],  # left
        [(x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1)],  # right
    ]
    return verts


def plot_geometry_3d(
    sim,
    *,
    show_domain: bool = True,
    show_ports: bool = True,
    title: str | None = None,
    figsize: tuple[float, float] = (10, 8),
    scale: float = 1e3,
    scale_label: str = "mm",
    backend: str = "auto",
) -> object:
    """Render simulation geometry in 3D.

    Parameters
    ----------
    sim : Simulation
        rfx Simulation object with geometry and ports defined.
    show_domain : bool
        Show the computational domain wireframe.
    show_ports : bool
        Show port locations as arrows.
    title : str or None
        Plot title.
    figsize : tuple
        Figure size.
    scale : float
        Coordinate scale factor (default 1e3 for mm).
    scale_label : str
        Unit label for axes.
    backend : str
        "auto" (pyvista if available, else matplotlib), "matplotlib", or "pyvista".

    Returns
    -------
    Figure (matplotlib) or Plotter (pyvista)
    """
    use_pyvista = (backend == "pyvista") or (backend == "auto" and HAS_PYVISTA)

    if use_pyvista and HAS_PYVISTA:
        return _plot_geometry_pyvista(sim, show_domain, show_ports, title, scale, scale_label)

    return _plot_geometry_mpl(sim, show_domain, show_ports, title, figsize, scale, scale_label)


def _plot_geometry_mpl(sim, show_domain, show_ports, title, figsize, scale, scale_label):
    """Matplotlib geometry renderer (3D if available, 2D multi-view fallback)."""
    _require_mpl()

    domain = sim._domain
    s = scale

    mat_colors = {
        "pec": ("gold", 0.6),
        "copper": ("orange", 0.6),
        "aluminum": ("silver", 0.5),
    }
    default_colors = [
        ("C0", 0.25), ("C1", 0.25), ("C2", 0.25),
        ("C3", 0.25), ("C4", 0.25),
    ]

    if HAS_MPL3D:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")

        color_idx = 0
        for entry in sim._geometry:
            mat_name = entry.material_name.lower()
            shape = entry.shape
            if mat_name in mat_colors:
                color, alpha = mat_colors[mat_name]
            else:
                color, alpha = default_colors[color_idx % len(default_colors)]
                color_idx += 1
            if hasattr(shape, "corner1") and hasattr(shape, "corner2"):
                p0 = tuple(c * s for c in shape.corner1)
                p1 = tuple(c * s for c in shape.corner2)
                faces = _box_faces(p0, p1)
                poly = Poly3DCollection(faces, alpha=alpha, facecolor=color,
                                        edgecolor="k", linewidth=0.5)
                ax.add_collection3d(poly)

        if show_domain:
            dx, dy, dz = [d * s for d in domain]
            for z in [0, dz]:
                ax.plot([0, dx, dx, 0, 0], [0, 0, dy, dy, 0], [z] * 5,
                        "k--", alpha=0.3, linewidth=0.5)
            for x, y in [(0, 0), (dx, 0), (dx, dy), (0, dy)]:
                ax.plot([x, x], [y, y], [0, dz], "k--", alpha=0.3, linewidth=0.5)

        if show_ports:
            axis_map = {"ex": 0, "ey": 1, "ez": 2}
            for pe in sim._ports:
                pos = [c * s for c in pe.position]
                direction = [0, 0, 0]
                ax_idx = axis_map.get(pe.component, 2)
                extent = (pe.extent or sim._domain[ax_idx] * 0.05) * s
                direction[ax_idx] = extent
                ax.quiver(*pos, *direction, color="red", arrow_length_ratio=0.3,
                          linewidth=2)

        ax.set_xlabel(f"x ({scale_label})")
        ax.set_ylabel(f"y ({scale_label})")
        ax.set_zlabel(f"z ({scale_label})")
        ax.set_title(title or "Simulation Geometry")

        dx, dy, dz = [d * s for d in domain]
        max_range = max(dx, dy, dz) / 2
        mid = [dx / 2, dy / 2, dz / 2]
        ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
        ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
        ax.set_zlim(mid[2] - max_range, mid[2] + max_range)
        fig.tight_layout()
        return fig

    # --- 2D multi-view fallback (XY, XZ, YZ projections) ---
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    view_labels = [
        ("x", "y", "XY (top)"),
        ("x", "z", "XZ (front)"),
        ("y", "z", "YZ (side)"),
    ]
    coord_map = {"x": 0, "y": 1, "z": 2}

    for ax, (h_label, v_label, vtitle) in zip(axes, view_labels):
        h_idx, v_idx = coord_map[h_label], coord_map[v_label]
        color_idx = 0

        for entry in sim._geometry:
            mat_name = entry.material_name.lower()
            shape = entry.shape
            if mat_name in mat_colors:
                color, alpha = mat_colors[mat_name]
            else:
                color, alpha = default_colors[color_idx % len(default_colors)]
                color_idx += 1
            if hasattr(shape, "corner1") and hasattr(shape, "corner2"):
                c1, c2 = shape.corner1, shape.corner2
                h0, h1 = c1[h_idx] * s, c2[h_idx] * s
                v0, v1 = c1[v_idx] * s, c2[v_idx] * s
                rect = plt.Rectangle((h0, v0), h1 - h0, v1 - v0,
                                      facecolor=color, alpha=alpha,
                                      edgecolor="k", linewidth=0.5)
                ax.add_patch(rect)

        if show_domain:
            dh, dv = domain[h_idx] * s, domain[v_idx] * s
            rect = plt.Rectangle((0, 0), dh, dv, fill=False,
                                  edgecolor="k", linestyle="--",
                                  alpha=0.3, linewidth=0.5)
            ax.add_patch(rect)

        if show_ports:
            axis_map = {"ex": 0, "ey": 1, "ez": 2}
            for pe in sim._ports:
                ph = pe.position[h_idx] * s
                pv = pe.position[v_idx] * s
                ax.plot(ph, pv, "rv", markersize=8)

        ax.set_xlabel(f"{h_label} ({scale_label})")
        ax.set_ylabel(f"{v_label} ({scale_label})")
        ax.set_title(vtitle)
        ax.set_aspect("equal")
        ax.autoscale()

    fig.suptitle(title or "Simulation Geometry", fontsize=14)
    fig.tight_layout()
    return fig


def _plot_geometry_pyvista(sim, show_domain, show_ports, title, scale, scale_label):
    """PyVista geometry renderer."""
    pl = pv.Plotter(off_screen=True)
    s = scale

    mat_colors = {
        "pec": "gold",
        "copper": "orange",
        "aluminum": "silver",
    }
    default_colors = ["steelblue", "coral", "mediumseagreen", "orchid", "sandybrown"]
    color_idx = 0

    for entry in sim._geometry:
        mat_name = entry.material_name.lower()
        shape = entry.shape

        color = mat_colors.get(mat_name)
        if color is None:
            color = default_colors[color_idx % len(default_colors)]
            color_idx += 1

        if hasattr(shape, "corner1") and hasattr(shape, "corner2"):
            p0 = [c * s for c in shape.corner1]
            p1 = [c * s for c in shape.corner2]
            bounds = [p0[0], p1[0], p0[1], p1[1], p0[2], p1[2]]
            box = pv.Box(bounds)
            opacity = 0.7 if mat_name == "pec" else 0.3
            pl.add_mesh(box, color=color, opacity=opacity, label=entry.material_name)

    if show_domain:
        dx, dy, dz = [d * s for d in sim._domain]
        domain_box = pv.Box([0, dx, 0, dy, 0, dz])
        pl.add_mesh(domain_box, style="wireframe", color="black", opacity=0.2)

    if show_ports:
        axis_map = {"ex": 0, "ey": 1, "ez": 2}
        for pe in sim._ports:
            pos = np.array([c * s for c in pe.position])
            ax_idx = axis_map.get(pe.component, 2)
            extent = (pe.extent or sim._domain[ax_idx] * 0.05) * s
            end = pos.copy()
            end[ax_idx] += extent
            line = pv.Line(pos, end)
            pl.add_mesh(line, color="red", line_width=4)

    pl.add_axes()
    pl.set_background("white")
    if title:
        pl.add_title(title)

    return pl


# ---------------------------------------------------------------------------
# Field visualization
# ---------------------------------------------------------------------------

def plot_field_3d(
    state,
    grid,
    *,
    component: str = "ez",
    threshold: float = 0.1,
    title: str | None = None,
    figsize: tuple[float, float] = (10, 8),
    cmap: str = "RdBu_r",
    scale: float = 1e3,
    scale_label: str = "mm",
    backend: str = "auto",
) -> object:
    """3D field visualization with isosurface or volume rendering.

    Parameters
    ----------
    state : FDTDState
    grid : Grid
    component : field component name
    threshold : float
        Isosurface threshold as fraction of peak value (0-1).
    title : str or None
    figsize : tuple
    cmap : colormap name
    scale : coordinate scale factor
    scale_label : unit label
    backend : "auto", "matplotlib", or "pyvista"

    Returns
    -------
    Figure (matplotlib) or Plotter (pyvista)
    """
    field = np.asarray(getattr(state, component))
    use_pyvista = (backend == "pyvista") or (backend == "auto" and HAS_PYVISTA)

    if use_pyvista and HAS_PYVISTA:
        return _plot_field_pyvista(field, grid, component, threshold,
                                   title, cmap, scale, scale_label)

    return _plot_field_mpl(field, grid, component, threshold,
                           title, figsize, cmap, scale, scale_label)


def _plot_field_mpl(field, grid, component, threshold, title, figsize, cmap, scale, scale_label):
    """Matplotlib field visualization (3D if available, else 2D tri-panel)."""
    _require_mpl()

    nx, ny, nz = field.shape
    dx = grid.dx * scale
    vmax = float(np.max(np.abs(field))) or 1.0

    if HAS_MPL3D:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")

        slices = [
            ("xy", field[:, :, nz // 2], 2, nz // 2),
            ("xz", field[:, ny // 2, :], 1, ny // 2),
            ("yz", field[nx // 2, :, :], 0, nx // 2),
        ]

        for name, slc, normal_axis, idx in slices:
            if normal_axis == 2:
                x = np.arange(slc.shape[0]) * dx
                y = np.arange(slc.shape[1]) * dx
                X, Y = np.meshgrid(x, y, indexing="ij")
                Z = np.full_like(X, idx * dx)
            elif normal_axis == 1:
                x = np.arange(slc.shape[0]) * dx
                z = np.arange(slc.shape[1]) * dx
                X, Z = np.meshgrid(x, z, indexing="ij")
                Y = np.full_like(X, idx * dx)
            else:
                y = np.arange(slc.shape[0]) * dx
                z = np.arange(slc.shape[1]) * dx
                Y, Z = np.meshgrid(y, z, indexing="ij")
                X = np.full_like(Y, idx * dx)

            colors = plt.get_cmap(cmap)((slc / vmax + 1) / 2)
            ax.plot_surface(X, Y, Z, facecolors=colors, alpha=0.5,
                            shade=False, rstride=1, cstride=1)

        ax.set_xlabel(f"x ({scale_label})")
        ax.set_ylabel(f"y ({scale_label})")
        ax.set_zlabel(f"z ({scale_label})")
        ax.set_title(title or f"{component} field (3 slices)")
        fig.tight_layout()
        return fig

    # --- 2D tri-panel fallback ---
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    slices_2d = [
        (field[:, :, nz // 2], "x", "y", f"z={nz // 2}"),
        (field[:, ny // 2, :], "x", "z", f"y={ny // 2}"),
        (field[nx // 2, :, :], "y", "z", f"x={nx // 2}"),
    ]

    for ax, (slc, xlabel, ylabel, slice_label) in zip(axes, slices_2d):
        im = ax.imshow(slc.T, origin="lower", cmap=cmap,
                        vmin=-vmax, vmax=vmax, aspect="equal",
                        extent=[0, slc.shape[0] * dx, 0, slc.shape[1] * dx])
        ax.set_xlabel(f"{xlabel} ({scale_label})")
        ax.set_ylabel(f"{ylabel} ({scale_label})")
        ax.set_title(f"{component} @ {slice_label}")
        fig.colorbar(im, ax=ax, shrink=0.6)

    fig.suptitle(title or f"{component} field (3 slices)", fontsize=14)
    fig.tight_layout()
    return fig


def _plot_field_pyvista(field, grid, component, threshold, title, cmap, scale, scale_label):
    """PyVista isosurface / volume rendering."""
    nx, ny, nz = field.shape
    dx = grid.dx * scale

    x = np.arange(nx + 1) * dx
    y = np.arange(ny + 1) * dx
    z = np.arange(nz + 1) * dx
    mesh = pv.RectilinearGrid(x, y, z)
    mesh.cell_data[component] = field.ravel(order="F")

    pl = pv.Plotter(off_screen=True)
    vmax = float(np.max(np.abs(field))) or 1.0

    # Volume rendering
    pl.add_volume(
        mesh,
        scalars=component,
        cmap=cmap,
        clim=[-vmax, vmax],
        opacity="sigmoid_5",
    )

    pl.add_axes()
    pl.set_background("white")
    if title:
        pl.add_title(title)

    return pl


# ---------------------------------------------------------------------------
# VTK export
# ---------------------------------------------------------------------------

def save_field_vtk(
    state,
    grid,
    filename: str,
    *,
    components: tuple[str, ...] = ("ex", "ey", "ez", "hx", "hy", "hz"),
    scale: float = 1e3,
) -> str:
    """Export field data to VTK format for ParaView.

    Parameters
    ----------
    state : FDTDState
    grid : Grid
    filename : str
        Output filename (without extension).
    components : tuple of str
        Field components to export.
    scale : float
        Coordinate scale factor.

    Returns
    -------
    str : path to the written file
    """
    nx, ny, nz = grid.shape
    dx = grid.dx * scale

    if HAS_PYVISTA:
        x = np.arange(nx + 1) * dx
        y = np.arange(ny + 1) * dx
        z = np.arange(nz + 1) * dx
        mesh = pv.RectilinearGrid(x, y, z)

        for comp in components:
            field = np.asarray(getattr(state, comp))
            mesh.cell_data[comp] = field.ravel(order="F")

        out_path = f"{filename}.vtk"
        mesh.save(out_path)
        return out_path

    # Fallback: write legacy VTK manually
    out_path = f"{filename}.vtk"
    with open(out_path, "w") as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("rfx FDTD field data\n")
        f.write("ASCII\n")
        f.write("DATASET RECTILINEAR_GRID\n")
        f.write(f"DIMENSIONS {nx + 1} {ny + 1} {nz + 1}\n")

        for axis_label, n in [("X", nx), ("Y", ny), ("Z", nz)]:
            coords = np.arange(n + 1) * dx
            f.write(f"{axis_label}_COORDINATES {n + 1} float\n")
            f.write(" ".join(f"{c:.6g}" for c in coords) + "\n")

        f.write(f"CELL_DATA {nx * ny * nz}\n")
        for comp in components:
            field = np.asarray(getattr(state, comp))
            f.write(f"SCALARS {comp} float 1\n")
            f.write("LOOKUP_TABLE default\n")
            # VTK rectilinear grid expects Fortran order
            for val in field.ravel(order="F"):
                f.write(f"{val:.6g}\n")

    return out_path


# ---------------------------------------------------------------------------
# Convenience: save geometry + field screenshot
# ---------------------------------------------------------------------------

def save_field_animation(
    snapshots: dict,
    grid,
    filename: str = "rfx_animation",
    *,
    component: str = "ez",
    slice_axis: str = "z",
    slice_index: int | None = None,
    fps: int = 15,
    cmap: str = "RdBu_r",
    scale: float = 1e3,
    dpi: int = 100,
) -> str:
    """Create a field animation (GIF/MP4) from snapshot data.

    Parameters
    ----------
    snapshots : dict
        From ``result.snapshots`` — keys are component names,
        values are (n_frames, ...) arrays.
    grid : Grid
    filename : str
        Output filename (without extension). GIF by default.
    component : str
        Field component to animate.
    slice_axis : "x", "y", or "z"
        Axis for 2D slice.
    slice_index : int or None
        Index along axis (default: center).
    fps : int
        Frames per second.
    cmap : colormap name.
    scale : coordinate scale factor.
    dpi : image resolution.

    Returns
    -------
    str : path to saved animation file.
    """
    _require_mpl()

    if component not in snapshots:
        raise ValueError(f"Component {component!r} not in snapshots: {list(snapshots.keys())}")

    data = np.asarray(snapshots[component])  # (n_frames, nx, ny) or (n_frames, ...)
    n_frames = data.shape[0]
    if n_frames == 0:
        raise ValueError("No frames in snapshot data")

    axis_idx = {"x": 0, "y": 1, "z": 2}[slice_axis]

    # If data is 3D per frame (full volume snapshots)
    if data.ndim == 4:
        if slice_index is None:
            slice_index = data.shape[axis_idx + 1] // 2
        if axis_idx == 0:
            data = data[:, slice_index, :, :]
        elif axis_idx == 1:
            data = data[:, :, slice_index, :]
        else:
            data = data[:, :, :, slice_index]

    # data is now (n_frames, n1, n2)
    vmax = float(np.max(np.abs(data))) or 1.0
    grid.dx * scale

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(data[0].T, origin="lower", cmap=cmap,
                   vmin=-vmax, vmax=vmax, aspect="equal")
    fig.colorbar(im, ax=ax, label=component)
    title = ax.set_title(f"{component} (frame 0/{n_frames})")

    axis_labels = {"x": ("y", "z"), "y": ("x", "z"), "z": ("x", "y")}
    xlabel, ylabel = axis_labels[slice_axis]
    ax.set_xlabel(f"{xlabel} (cells)")
    ax.set_ylabel(f"{ylabel} (cells)")

    def _update(frame):
        im.set_data(data[frame].T)
        title.set_text(f"{component} (frame {frame}/{n_frames})")
        return [im, title]

    try:
        from matplotlib.animation import FuncAnimation, PillowWriter
        anim = FuncAnimation(fig, _update, frames=n_frames, blit=True, interval=1000//fps)
        out_path = f"{filename}.gif"
        anim.save(out_path, writer=PillowWriter(fps=fps), dpi=dpi)
        plt.close(fig)
        return out_path
    except Exception:
        # Fallback: save individual frames
        import os
        frame_dir = f"{filename}_frames"
        os.makedirs(frame_dir, exist_ok=True)
        for i in range(min(n_frames, 100)):
            im.set_data(data[i].T)
            title.set_text(f"{component} (frame {i}/{n_frames})")
            fig.savefig(f"{frame_dir}/frame_{i:04d}.png", dpi=dpi)
        plt.close(fig)
        return frame_dir


def save_screenshot(
    sim,
    state=None,
    filename: str = "rfx_3d",
    *,
    component: str = "ez",
    dpi: int = 150,
    figsize: tuple[float, float] = (10, 8),
    scale: float = 1e3,
) -> str:
    """Save a 3D screenshot of geometry (and optionally field).

    Parameters
    ----------
    sim : Simulation
    state : FDTDState or None
        If provided, overlays field slices on geometry.
    filename : str
        Output filename (without extension).
    component : field component for overlay
    dpi : image resolution
    figsize : figure size
    scale : coordinate scale

    Returns
    -------
    str : path to saved image
    """
    if state is not None:
        fig = plot_field_3d(state, sim._build_grid(), component=component,
                            figsize=figsize, scale=scale, backend="matplotlib")
    else:
        fig = plot_geometry_3d(sim, figsize=figsize, scale=scale,
                               backend="matplotlib")

    out_path = f"{filename}.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path
