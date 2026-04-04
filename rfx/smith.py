"""Smith chart plotting for S-parameter visualization.

Draws the standard Smith chart grid (constant-resistance and
constant-reactance circles) and plots S11 data as a trajectory.
Follows the same conventions as :mod:`rfx.visualize`.
"""

from __future__ import annotations

import numpy as np

try:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Arc
    from matplotlib.lines import Line2D
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def _require_mpl():
    if not HAS_MPL:
        raise ImportError("matplotlib is required for Smith chart plotting")


def _draw_smith_grid(ax):
    """Draw constant-r and constant-x circles on *ax*.

    The Smith chart lives inside the unit circle in the reflection-coefficient
    plane.  Constant-resistance circles have centre ``(r/(r+1), 0)`` and
    radius ``1/(r+1)``.  Constant-reactance arcs have centre ``(1, 1/x)``
    and radius ``1/|x|``.
    """
    # Outer boundary (|Gamma| = 1)
    boundary = Circle((0, 0), 1, fill=False, edgecolor="black",
                       linewidth=1.0, zorder=1)
    ax.add_patch(boundary)

    # --- constant-resistance circles ---
    r_values = [0, 0.2, 0.5, 1, 2, 5]
    for r in r_values:
        cx = r / (r + 1)
        cr = 1 / (r + 1)
        circ = Circle((cx, 0), cr, fill=False, edgecolor="0.70",
                       linewidth=0.5, zorder=0)
        ax.add_patch(circ)
        # Clip to the unit circle is handled by axis limits + clip_on

    # --- constant-reactance arcs ---
    x_values = [0.2, 0.5, 1, 2, 5]
    for x in x_values:
        radius = 1 / abs(x)
        # Positive reactance (upper half)
        _draw_reactance_arc(ax, x, radius)
        # Negative reactance (lower half)
        _draw_reactance_arc(ax, -x, radius)

    # Real axis
    ax.plot([-1, 1], [0, 0], color="0.70", linewidth=0.5, zorder=0)


def _draw_reactance_arc(ax, x, radius):
    """Draw a single constant-reactance arc clipped to the unit circle."""
    # Centre of the reactance circle is at (1, 1/x), radius 1/|x|.
    cy = 1 / x
    cx = 1.0

    # Parametrise and clip to unit disk
    theta = np.linspace(0, 2 * np.pi, 300)
    pts_x = cx + radius * np.cos(theta)
    pts_y = cy + radius * np.sin(theta)

    # Keep only points inside (or on) the unit circle
    mask = pts_x**2 + pts_y**2 <= 1.0 + 1e-6
    if not np.any(mask):
        return

    ax.plot(pts_x[mask], pts_y[mask], color="0.70", linewidth=0.5, zorder=0)


def _draw_vswr_circle(ax, vswr, **kwargs):
    """Draw a constant-VSWR circle centred at the origin."""
    if vswr < 1:
        return
    gamma_mag = (vswr - 1) / (vswr + 1)
    style = dict(fill=False, edgecolor="blue", linewidth=0.8,
                 linestyle="--", alpha=0.5, zorder=1)
    style.update(kwargs)
    circ = Circle((0, 0), gamma_mag, **style)
    ax.add_patch(circ)


def plot_smith(
    s11: np.ndarray,
    freqs: np.ndarray,
    *,
    z0: float = 50.0,
    ax=None,
    show_vswr: bool = True,
    markers: list[float] | None = None,
    title: str | None = None,
):
    """Plot S11 data on a Smith chart.

    Parameters
    ----------
    s11 : (n_freqs,) complex
        Reflection coefficient (Gamma) values.
    freqs : (n_freqs,) Hz
        Corresponding frequencies.
    z0 : float
        Reference impedance (used only for annotation, not for
        re-normalisation).
    ax : matplotlib Axes, optional
        Axes to plot on.  If *None*, a new figure is created.
    show_vswr : bool
        If True, draw VSWR = 2 and VSWR = 3 circles.
    markers : list of float, optional
        Frequencies (in Hz) at which to place labelled markers.
    title : str, optional
        Plot title.  Defaults to ``"Smith Chart"``.

    Returns
    -------
    matplotlib Axes
    """
    _require_mpl()

    s11 = np.asarray(s11)
    freqs = np.asarray(freqs)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        pass

    # Draw the grid
    _draw_smith_grid(ax)

    # VSWR circles
    if show_vswr:
        _draw_vswr_circle(ax, 2.0)
        _draw_vswr_circle(ax, 3.0)

    # Plot S11 trajectory
    ax.plot(s11.real, s11.imag, color="C0", linewidth=1.5, zorder=3)

    # Start / end dots
    ax.plot(s11.real[0], s11.imag[0], "o", color="C0", markersize=5, zorder=4)
    ax.plot(s11.real[-1], s11.imag[-1], "s", color="C0", markersize=5, zorder=4)

    # Frequency markers
    if markers is not None:
        for f_mark in markers:
            idx = int(np.argmin(np.abs(freqs - f_mark)))
            gamma = s11[idx]
            f_ghz = freqs[idx] / 1e9
            ax.plot(gamma.real, gamma.imag, "D", color="C3", markersize=7,
                    zorder=5)
            ax.annotate(
                f"{f_ghz:.2f} GHz",
                (gamma.real, gamma.imag),
                textcoords="offset points",
                xytext=(8, 8),
                fontsize=8,
                color="C3",
                zorder=5,
            )

    # Appearance
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.15, 1.15)
    ax.set_aspect("equal")
    ax.set_title(title or "Smith Chart")
    ax.axis("off")

    return ax
