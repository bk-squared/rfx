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
