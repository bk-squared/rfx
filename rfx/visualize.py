"""Field visualization utilities.

Provides matplotlib-based plotting for field snapshots, S-parameters,
and radiation patterns.  All functions return the figure for further
customization or saving.
"""

from __future__ import annotations

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
