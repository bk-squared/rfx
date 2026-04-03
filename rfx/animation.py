"""Field animation export (MP4/GIF).

Uses matplotlib.animation for rendering. Requires Pillow for GIF,
ffmpeg for MP4.
"""

from __future__ import annotations

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def _require_mpl():
    if not HAS_MPL:
        raise ImportError(
            "matplotlib and Pillow are required for animation export. "
            "Install with: pip install matplotlib Pillow"
        )


def save_field_animation(
    result,
    filename: str,
    *,
    component: str = "ez",
    slice_axis: str = "z",
    slice_index: int | None = None,
    fps: int = 15,
    colormap: str = "RdBu_r",
    vmin: float | None = None,
    vmax: float | None = None,
    dpi: int = 100,
    figsize: tuple = (6, 5),
    interval: int = 1,
):
    """Save field animation from simulation snapshots.

    Parameters
    ----------
    result : rfx.Result or dict
        Simulation result with ``.snapshots`` attribute, or a raw
        snapshots dict mapping component names to arrays of shape
        ``(n_frames, nx, ny, nz)`` or ``(n_frames, nx, ny)``.
    filename : str
        Output file path. Extension determines format:
        ``.gif`` uses PillowWriter, ``.mp4`` uses FFMpegWriter
        (falls back to GIF if ffmpeg is unavailable).
        If no recognised extension, ``.gif`` is appended.
    component : str
        Field component to animate (default ``"ez"``).
    slice_axis : str
        Axis normal to the slice plane (``"x"``, ``"y"``, or ``"z"``).
        Only used when snapshot data is 3D per frame.
    slice_index : int or None
        Index along *slice_axis*. ``None`` selects the centre.
    fps : int
        Frames per second.
    colormap : str
        Matplotlib colormap name.
    vmin, vmax : float or None
        Colour-scale limits. ``None`` computes symmetric auto-range
        from the data.
    dpi : int
        Resolution in dots per inch.
    figsize : tuple
        Figure size in inches ``(width, height)``.
    interval : int
        Frame stride — use every *interval*-th snapshot frame.
        Default 1 (every frame).

    Returns
    -------
    str
        Path to the saved animation file.

    Raises
    ------
    ValueError
        If *result* has no snapshots or the requested *component*
        is missing.
    ImportError
        If matplotlib or Pillow are not installed.
    """
    _require_mpl()

    # ------------------------------------------------------------------
    # Extract snapshot data
    # ------------------------------------------------------------------
    if isinstance(result, dict):
        snapshots = result
    elif hasattr(result, "snapshots"):
        snapshots = result.snapshots
    else:
        raise ValueError(
            "result must be an rfx.Result with .snapshots or a dict "
            "mapping component names to arrays"
        )

    if snapshots is None or len(snapshots) == 0:
        raise ValueError(
            "No snapshots found. Run the simulation with "
            "snapshot=SnapshotSpec(...) to record field snapshots."
        )

    if component not in snapshots:
        raise ValueError(
            f"Component {component!r} not in snapshots. "
            f"Available: {list(snapshots.keys())}"
        )

    data = np.asarray(snapshots[component])

    # ------------------------------------------------------------------
    # Apply frame stride
    # ------------------------------------------------------------------
    if interval > 1:
        data = data[::interval]

    n_frames = data.shape[0]
    if n_frames == 0:
        raise ValueError("No frames in snapshot data")

    # ------------------------------------------------------------------
    # Slice 3D volumes to 2D
    # ------------------------------------------------------------------
    axis_map = {"x": 0, "y": 1, "z": 2}
    if slice_axis not in axis_map:
        raise ValueError(
            f"slice_axis must be 'x', 'y', or 'z', got {slice_axis!r}"
        )
    axis_idx = axis_map[slice_axis]

    if data.ndim == 4:
        # (n_frames, nx, ny, nz) — full volume snapshots
        if slice_index is None:
            slice_index = data.shape[axis_idx + 1] // 2
        if axis_idx == 0:
            data = data[:, slice_index, :, :]
        elif axis_idx == 1:
            data = data[:, :, slice_index, :]
        else:
            data = data[:, :, :, slice_index]
    elif data.ndim == 3:
        # (n_frames, n1, n2) — already 2D sliced, nothing to do
        pass
    else:
        raise ValueError(
            f"Unexpected snapshot shape {data.shape}; expected "
            f"(n_frames, nx, ny, nz) or (n_frames, n1, n2)"
        )

    # ------------------------------------------------------------------
    # Determine colour limits
    # ------------------------------------------------------------------
    if vmax is None:
        vmax = float(np.max(np.abs(data))) or 1.0
    if vmin is None:
        vmin = -vmax

    # ------------------------------------------------------------------
    # Build animation
    # ------------------------------------------------------------------
    axis_labels = {"x": ("y", "z"), "y": ("x", "z"), "z": ("x", "y")}
    xlabel, ylabel = axis_labels[slice_axis]

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(
        data[0].T,
        origin="lower",
        cmap=colormap,
        vmin=vmin,
        vmax=vmax,
        aspect="equal",
    )
    fig.colorbar(im, ax=ax, label=component)
    title_obj = ax.set_title(f"{component} (frame 1/{n_frames})")
    ax.set_xlabel(f"{xlabel} (cells)")
    ax.set_ylabel(f"{ylabel} (cells)")

    def _update(frame):
        im.set_data(data[frame].T)
        title_obj.set_text(f"{component} (frame {frame + 1}/{n_frames})")
        return [im, title_obj]

    anim = FuncAnimation(
        fig, _update, frames=n_frames, blit=True, interval=1000 // fps,
    )

    # ------------------------------------------------------------------
    # Determine output format and save
    # ------------------------------------------------------------------
    filename = str(filename)
    lower = filename.lower()

    if lower.endswith(".mp4"):
        try:
            from matplotlib.animation import FFMpegWriter
            writer = FFMpegWriter(fps=fps)
        except (ImportError, RuntimeError):
            # ffmpeg not available — fall back to GIF
            filename = filename[:-4] + ".gif"
            writer = PillowWriter(fps=fps)
    elif lower.endswith(".gif"):
        writer = PillowWriter(fps=fps)
    else:
        # Default to GIF
        if not lower.endswith(".gif"):
            filename = filename + ".gif"
        writer = PillowWriter(fps=fps)

    anim.save(filename, writer=writer, dpi=dpi)
    plt.close(fig)

    return filename
