"""Reusable Streamlit UI components for the rfx dashboard.

Separates widget logic from the main app so individual panels can be
tested and reused independently.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Streamlit is imported lazily so the module can be imported without it
# (e.g. for testing or programmatic use).
# ---------------------------------------------------------------------------

_st = None


def _require_st():
    """Import streamlit on first use and cache it."""
    global _st
    if _st is None:
        try:
            import streamlit as st
            _st = st
        except ImportError:
            raise ImportError(
                "streamlit is required for the dashboard. "
                "Install with: pip install rfx-fdtd[dashboard]"
            )
    return _st


# ---------------------------------------------------------------------------
# Data containers for collecting UI state
# ---------------------------------------------------------------------------

@dataclass
class GeometryEntry:
    """One user-defined geometry item."""
    shape_type: str  # "Box", "Sphere", "Cylinder"
    material: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class PortEntry:
    """One user-defined port."""
    position: tuple[float, float, float]
    component: str
    impedance: float
    f0_ghz: float
    bandwidth: float


@dataclass
class ProbeEntry:
    """One user-defined field probe."""
    position: tuple[float, float, float]
    component: str


# ---------------------------------------------------------------------------
# Geometry builder panel
# ---------------------------------------------------------------------------

def geometry_panel(materials: list[str]) -> list[GeometryEntry]:
    """Render the geometry builder and return the list of entries.

    Uses ``st.session_state["geometries"]`` for persistence across reruns.
    """
    st = _require_st()

    if "geometries" not in st.session_state:
        st.session_state["geometries"] = []

    st.subheader("Add Geometry")

    col1, col2 = st.columns([1, 1])
    with col1:
        shape_type = st.selectbox("Shape", ["Box", "Sphere", "Cylinder"], key="geo_shape")
    with col2:
        mat = st.selectbox("Material", materials, key="geo_mat")

    params: dict[str, Any] = {}

    if shape_type == "Box":
        st.caption("Define two corners (mm)")
        c1, c2, c3 = st.columns(3)
        with c1:
            x_lo = st.number_input("x_lo (mm)", value=0.0, step=1.0, key="box_xlo")
            x_hi = st.number_input("x_hi (mm)", value=50.0, step=1.0, key="box_xhi")
        with c2:
            y_lo = st.number_input("y_lo (mm)", value=0.0, step=1.0, key="box_ylo")
            y_hi = st.number_input("y_hi (mm)", value=50.0, step=1.0, key="box_yhi")
        with c3:
            z_lo = st.number_input("z_lo (mm)", value=0.0, step=0.1, key="box_zlo")
            z_hi = st.number_input("z_hi (mm)", value=1.6, step=0.1, key="box_zhi")
        params = {
            "corner_lo": (x_lo * 1e-3, y_lo * 1e-3, z_lo * 1e-3),
            "corner_hi": (x_hi * 1e-3, y_hi * 1e-3, z_hi * 1e-3),
        }

    elif shape_type == "Sphere":
        st.caption("Center (mm) and radius (mm)")
        c1, c2 = st.columns(2)
        with c1:
            cx = st.number_input("center_x (mm)", value=25.0, step=1.0, key="sph_cx")
            cy = st.number_input("center_y (mm)", value=25.0, step=1.0, key="sph_cy")
            cz = st.number_input("center_z (mm)", value=12.5, step=1.0, key="sph_cz")
        with c2:
            r = st.number_input("radius (mm)", value=10.0, step=1.0, min_value=0.1, key="sph_r")
        params = {
            "center": (cx * 1e-3, cy * 1e-3, cz * 1e-3),
            "radius": r * 1e-3,
        }

    elif shape_type == "Cylinder":
        st.caption("Center (mm), radius (mm), height (mm), axis")
        c1, c2 = st.columns(2)
        with c1:
            cx = st.number_input("center_x (mm)", value=25.0, step=1.0, key="cyl_cx")
            cy = st.number_input("center_y (mm)", value=25.0, step=1.0, key="cyl_cy")
            cz = st.number_input("center_z (mm)", value=12.5, step=1.0, key="cyl_cz")
        with c2:
            r = st.number_input("radius (mm)", value=5.0, step=1.0, min_value=0.1, key="cyl_r")
            h = st.number_input("height (mm)", value=10.0, step=1.0, min_value=0.1, key="cyl_h")
            axis = st.selectbox("axis", ["z", "x", "y"], key="cyl_axis")
        params = {
            "center": (cx * 1e-3, cy * 1e-3, cz * 1e-3),
            "radius": r * 1e-3,
            "height": h * 1e-3,
            "axis": axis,
        }

    if st.button("Add Shape", key="add_geo"):
        entry = GeometryEntry(shape_type=shape_type, material=mat, params=params)
        st.session_state["geometries"].append(entry)
        st.success(f"Added {shape_type} ({mat})")

    # Display current geometry list
    geos: list[GeometryEntry] = st.session_state["geometries"]
    if geos:
        st.markdown("---")
        st.caption(f"**{len(geos)} shape(s) defined**")
        for i, g in enumerate(geos):
            label = f"{i + 1}. {g.shape_type} — {g.material}"
            col_label, col_btn = st.columns([4, 1])
            with col_label:
                st.text(label)
            with col_btn:
                if st.button("Remove", key=f"rm_geo_{i}"):
                    st.session_state["geometries"].pop(i)
                    st.rerun()

    return geos


# ---------------------------------------------------------------------------
# Source / port panel
# ---------------------------------------------------------------------------

def source_panel() -> tuple[list[PortEntry], list[ProbeEntry]]:
    """Render source and probe configuration widgets.

    Returns (ports, probes).
    """
    st = _require_st()

    if "ports" not in st.session_state:
        st.session_state["ports"] = []
    if "probes" not in st.session_state:
        st.session_state["probes"] = []

    # --- Ports ---
    st.subheader("Ports")
    st.caption("Add a lumped excitation port")
    c1, c2, c3 = st.columns(3)
    with c1:
        px = st.number_input("port x (mm)", value=25.0, step=1.0, key="port_x")
        py = st.number_input("port y (mm)", value=25.0, step=1.0, key="port_y")
        pz = st.number_input("port z (mm)", value=1.6, step=0.1, key="port_z")
    with c2:
        comp = st.selectbox("component", ["ez", "ex", "ey"], key="port_comp")
        z0 = st.number_input("impedance (ohm)", value=50.0, step=10.0, key="port_z0")
    with c3:
        f0 = st.number_input("f0 (GHz)", value=2.5, step=0.5, key="port_f0")
        bw = st.number_input("bandwidth", value=0.8, step=0.1, min_value=0.1, max_value=1.0, key="port_bw")

    if st.button("Add Port", key="add_port"):
        entry = PortEntry(
            position=(px * 1e-3, py * 1e-3, pz * 1e-3),
            component=comp, impedance=z0,
            f0_ghz=f0, bandwidth=bw,
        )
        st.session_state["ports"].append(entry)
        st.success(f"Added port at ({px}, {py}, {pz}) mm")

    ports: list[PortEntry] = st.session_state["ports"]
    for i, p in enumerate(ports):
        col_l, col_b = st.columns([4, 1])
        with col_l:
            st.text(f"Port {i + 1}: ({p.position[0]*1e3:.1f}, {p.position[1]*1e3:.1f}, {p.position[2]*1e3:.1f}) mm  {p.component}  Z={p.impedance:.0f} ohm")
        with col_b:
            if st.button("Remove", key=f"rm_port_{i}"):
                st.session_state["ports"].pop(i)
                st.rerun()

    # --- Probes ---
    st.markdown("---")
    st.subheader("Field Probes")
    st.caption("Add a field probe to record time-domain data")
    c1, c2 = st.columns(2)
    with c1:
        prx = st.number_input("probe x (mm)", value=30.0, step=1.0, key="probe_x")
        pry = st.number_input("probe y (mm)", value=30.0, step=1.0, key="probe_y")
        prz = st.number_input("probe z (mm)", value=1.6, step=0.1, key="probe_z")
    with c2:
        pr_comp = st.selectbox("component", ["ez", "ex", "ey", "hx", "hy", "hz"], key="probe_comp")

    if st.button("Add Probe", key="add_probe"):
        entry = ProbeEntry(
            position=(prx * 1e-3, pry * 1e-3, prz * 1e-3),
            component=pr_comp,
        )
        st.session_state["probes"].append(entry)
        st.success(f"Added probe at ({prx}, {pry}, {prz}) mm")

    probes: list[ProbeEntry] = st.session_state["probes"]
    for i, pr in enumerate(probes):
        col_l, col_b = st.columns([4, 1])
        with col_l:
            st.text(f"Probe {i + 1}: ({pr.position[0]*1e3:.1f}, {pr.position[1]*1e3:.1f}, {pr.position[2]*1e3:.1f}) mm  {pr.component}")
        with col_b:
            if st.button("Remove", key=f"rm_probe_{i}"):
                st.session_state["probes"].pop(i)
                st.rerun()

    return ports, probes


# ---------------------------------------------------------------------------
# Results display helpers
# ---------------------------------------------------------------------------

def display_s_params(s_params: np.ndarray, freqs: np.ndarray) -> None:
    """Plot S-parameter magnitude (dB) using Streamlit's built-in chart."""
    st = _require_st()

    n_ports = s_params.shape[0]
    freqs_ghz = freqs / 1e9

    # Try matplotlib first (better quality)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 4.5))
        for i in range(n_ports):
            for j in range(n_ports):
                mag_db = 20 * np.log10(np.maximum(np.abs(s_params[i, j, :]), 1e-10))
                ax.plot(freqs_ghz, mag_db, label=f"S{i+1}{j+1}")
        ax.set_xlabel("Frequency (GHz)")
        ax.set_ylabel("Magnitude (dB)")
        ax.set_title("S-Parameters")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    except ImportError:
        # Fallback: line chart via Streamlit
        import pandas as pd
        data = {"Frequency (GHz)": freqs_ghz}
        for i in range(n_ports):
            for j in range(n_ports):
                mag_db = 20 * np.log10(np.maximum(np.abs(s_params[i, j, :]), 1e-10))
                data[f"S{i+1}{j+1} (dB)"] = mag_db
        df = pd.DataFrame(data).set_index("Frequency (GHz)")
        st.line_chart(df)


def display_smith_chart(s11: np.ndarray, freqs: np.ndarray) -> None:
    """Render Smith chart for S11 data."""
    st = _require_st()
    try:
        import matplotlib
        matplotlib.use("Agg")
        from rfx.smith import plot_smith
        fig = plot_smith(s11, freqs)
        st.pyplot(fig)
        import matplotlib.pyplot as plt
        plt.close(fig)
    except ImportError:
        st.info("Smith chart requires matplotlib. Install with: pip install matplotlib")


def display_field_slice(state, grid, component: str = "ez", axis: str = "z") -> None:
    """Show a 2D field slice as an image."""
    st = _require_st()
    try:
        import matplotlib
        matplotlib.use("Agg")
        from rfx.visualize import plot_field_slice
        fig = plot_field_slice(state, grid, component=component, axis=axis)
        st.pyplot(fig)
        import matplotlib.pyplot as plt
        plt.close(fig)
    except ImportError:
        # Fallback: raw array via st.image
        field = np.asarray(getattr(state, component))
        axis_idx = {"x": 0, "y": 1, "z": 2}[axis]
        idx = field.shape[axis_idx] // 2
        if axis_idx == 0:
            slc = field[idx, :, :]
        elif axis_idx == 1:
            slc = field[:, idx, :]
        else:
            slc = field[:, :, idx]
        vmax = float(np.max(np.abs(slc))) or 1.0
        normalized = (slc / vmax * 127 + 128).clip(0, 255).astype(np.uint8)
        st.image(normalized, caption=f"{component} slice ({axis})", use_container_width=True)


def display_time_series(time_series: np.ndarray, dt: float) -> None:
    """Plot probe time series."""
    st = _require_st()
    ts = np.asarray(time_series)
    if ts.ndim == 1:
        ts = ts[:, np.newaxis]
    n_steps = ts.shape[0]

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 3.5))
        t_ns = np.arange(n_steps) * dt * 1e9
        for i in range(ts.shape[1]):
            ax.plot(t_ns, ts[:, i], lw=0.8, label=f"Probe {i + 1}")
        ax.set_xlabel("Time (ns)")
        ax.set_ylabel("Amplitude")
        ax.set_title("Probe Time Series")
        if ts.shape[1] > 1:
            ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    except ImportError:
        import pandas as pd
        t_ns = np.arange(n_steps) * dt * 1e9
        data = {"Time (ns)": t_ns}
        for i in range(ts.shape[1]):
            data[f"Probe {i + 1}"] = ts[:, i]
        df = pd.DataFrame(data).set_index("Time (ns)")
        st.line_chart(df)


def export_touchstone(s_params: np.ndarray, freqs: np.ndarray) -> bytes:
    """Generate Touchstone file content as bytes for download."""
    import tempfile
    from pathlib import Path

    n_ports = s_params.shape[0]
    suffix = f".s{n_ports}p"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        from rfx.io import write_touchstone
        write_touchstone(tmp_path, s_params, freqs)
        return tmp_path.read_bytes()
    finally:
        tmp_path.unlink(missing_ok=True)
