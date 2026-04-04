"""rfx Web Dashboard — Streamlit-based FDTD simulation interface.

Run with::

    streamlit run rfx/dashboard/app.py

Provides an interactive UI for configuring and running rfx FDTD
simulations without writing code.  Results include S-parameter plots,
Smith charts, field snapshots, and Touchstone export.
"""

from __future__ import annotations

import time

import numpy as np
import streamlit as st

from rfx.dashboard.components import (
    GeometryEntry,
    PortEntry,
    ProbeEntry,
    display_field_slice,
    display_s_params,
    display_smith_chart,
    display_time_series,
    export_touchstone,
    geometry_panel,
    source_panel,
)

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="rfx FDTD Simulator",
    page_icon="📡",
    layout="wide",
)

try:
    from rfx import __version__ as _rfx_version
except ImportError:
    _rfx_version = "unknown"

st.title("rfx — JAX FDTD Simulator")
st.caption(f"Interactive electromagnetic simulation powered by rfx v{_rfx_version}")

# ---------------------------------------------------------------------------
# Material library (populated from rfx)
# ---------------------------------------------------------------------------

try:
    from rfx.api import MATERIAL_LIBRARY
    _AVAILABLE_MATERIALS = sorted(MATERIAL_LIBRARY.keys())
except ImportError:
    _AVAILABLE_MATERIALS = [
        "vacuum", "air", "fr4", "rogers4003c", "alumina",
        "silicon", "ptfe", "copper", "aluminum", "pec",
    ]

# ---------------------------------------------------------------------------
# Sidebar: simulation parameters
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Simulation Setup")

    st.subheader("Frequency range")
    col_fmin, col_fmax = st.columns(2)
    with col_fmin:
        freq_min = st.number_input(
            "f_min [GHz]", value=1.0, step=0.5, min_value=0.1, key="freq_min",
        )
    with col_fmax:
        freq_max = st.number_input(
            "f_max [GHz]", value=4.0, step=0.5, min_value=0.1, key="freq_max",
        )
    if freq_min >= freq_max:
        st.warning("f_min must be less than f_max.")

    st.subheader("Domain size")
    col_dx, col_dy, col_dz = st.columns(3)
    with col_dx:
        dom_x = st.number_input("Lx [mm]", value=50.0, step=5.0, min_value=1.0, key="dom_x")
    with col_dy:
        dom_y = st.number_input("Ly [mm]", value=50.0, step=5.0, min_value=1.0, key="dom_y")
    with col_dz:
        dom_z = st.number_input("Lz [mm]", value=25.0, step=5.0, min_value=1.0, key="dom_z")

    st.subheader("Solver")
    boundary = st.selectbox("Boundary Condition", ["cpml", "pec"], key="boundary")
    accuracy = st.selectbox("Accuracy", ["draft", "standard", "high"], key="accuracy")
    mode = st.selectbox("Mode", ["3d", "2d_tmz", "2d_tez"], key="mode")

    accuracy_map = {
        "draft": {"num_periods": 10.0, "cells_per_wl": 10},
        "standard": {"num_periods": 20.0, "cells_per_wl": 20},
        "high": {"num_periods": 40.0, "cells_per_wl": 30},
    }
    acc = accuracy_map[accuracy]

    st.markdown("---")
    st.subheader("Advanced")
    n_steps_override = st.number_input(
        "Override n_steps (0 = auto)", value=0, step=100, min_value=0, key="n_steps",
    )
    cpml_layers = st.number_input(
        "CPML layers", value=12, step=2, min_value=4, max_value=30, key="cpml_layers",
    )

    st.markdown("---")
    st.caption(f"rfx v{_rfx_version} — JAX-native 3D FDTD")

# ---------------------------------------------------------------------------
# Main area: tabs
# ---------------------------------------------------------------------------

tab_geo, tab_source, tab_results = st.tabs(
    ["Geometry & Materials", "Sources & Ports", "Results"],
)

# --- Geometry tab ---
with tab_geo:
    geometries = geometry_panel(_AVAILABLE_MATERIALS)

    # Material reference
    with st.expander("Material Library Reference"):
        try:
            from rfx.api import MATERIAL_LIBRARY as _ml
            rows = []
            for name, props in sorted(_ml.items()):
                eps_r = props.get("eps_r", 1.0)
                sigma = props.get("sigma", 0.0)
                has_disp = "debye_poles" in props or "lorentz_poles" in props
                rows.append({
                    "Name": name,
                    "eps_r": f"{eps_r:.2f}",
                    "sigma (S/m)": f"{sigma:.2e}",
                    "Dispersive": "Yes" if has_disp else "No",
                })
            st.table(rows)
        except ImportError:
            st.info("Install rfx to see material properties.")

# --- Source tab ---
with tab_source:
    ports, probes = source_panel()

# --- Results tab ---
with tab_results:
    st.subheader("Simulation Results")

    # Summary of what will be simulated
    n_geo = len(st.session_state.get("geometries", []))
    n_ports = len(st.session_state.get("ports", []))
    n_probes = len(st.session_state.get("probes", []))

    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1:
        st.metric("Shapes", n_geo)
    with col_s2:
        st.metric("Ports", n_ports)
    with col_s3:
        st.metric("Probes", n_probes)

    # Validation
    can_run = True
    if freq_min >= freq_max:
        st.error("Invalid frequency range: min >= max.")
        can_run = False
    if n_ports == 0 and n_probes == 0:
        st.warning("Add at least one port or probe to capture results.")
        can_run = False

    run_clicked = st.button(
        "Run Simulation", type="primary", disabled=not can_run, key="run_sim",
    )

    if run_clicked:
        try:
            from rfx import Simulation, Box, Sphere, Cylinder
            from rfx.sources.sources import GaussianPulse
        except ImportError:
            st.error(
                "rfx is not installed or JAX is unavailable. "
                "Install with: pip install -e .[all]"
            )
            st.stop()

        # --- Build simulation ---
        freq_max_hz = freq_max * 1e9
        freq_min_hz = freq_min * 1e9
        domain_m = (dom_x * 1e-3, dom_y * 1e-3, dom_z * 1e-3)

        # Cell size from accuracy
        from rfx.grid import C0
        wavelength_min = C0 / freq_max_hz
        dx = wavelength_min / acc["cells_per_wl"]

        sim = Simulation(
            freq_max=freq_max_hz,
            domain=domain_m,
            boundary=boundary,
            cpml_layers=cpml_layers if boundary == "cpml" else 0,
            dx=dx,
            mode=mode,
        )

        # Add geometries
        shape_builders = {
            "Box": lambda p: Box(p["corner_lo"], p["corner_hi"]),
            "Sphere": lambda p: Sphere(p["center"], p["radius"]),
            "Cylinder": lambda p: Cylinder(
                p["center"], p["radius"], p["height"], p.get("axis", "z"),
            ),
        }
        geos: list[GeometryEntry] = st.session_state.get("geometries", [])
        for g in geos:
            builder = shape_builders.get(g.shape_type)
            if builder:
                shape = builder(g.params)
                sim.add(shape, material=g.material)

        # Add ports
        port_entries: list[PortEntry] = st.session_state.get("ports", [])
        for p in port_entries:
            waveform = GaussianPulse(
                f0=p.f0_ghz * 1e9,
                bandwidth=p.bandwidth,
            )
            sim.add_port(
                p.position,
                p.component,
                impedance=p.impedance,
                waveform=waveform,
            )

        # Add probes
        probe_entries: list[ProbeEntry] = st.session_state.get("probes", [])
        for pr in probe_entries:
            sim.add_probe(pr.position, pr.component)

        # --- Run ---
        n_steps_val = n_steps_override if n_steps_override > 0 else None
        num_periods = acc["num_periods"]

        progress_bar = st.progress(0, text="Initializing simulation...")
        t_start = time.time()

        try:
            progress_bar.progress(10, text="Building grid and materials...")

            # Determine actual step count for progress display
            if n_steps_val is None:
                grid_preview = sim._build_grid()
                actual_steps = grid_preview.num_timesteps(num_periods=num_periods)
            else:
                actual_steps = n_steps_val

            st.info(f"Running {actual_steps} timesteps...")
            progress_bar.progress(20, text=f"Running FDTD ({actual_steps} steps)...")

            result = sim.run(
                n_steps=n_steps_val,
                num_periods=num_periods,
                compute_s_params=True if port_entries else None,
            )

            elapsed = time.time() - t_start
            progress_bar.progress(100, text="Complete!")
            st.success(f"Simulation complete in {elapsed:.1f}s ({actual_steps} steps)")

        except Exception as exc:
            progress_bar.empty()
            st.error(f"Simulation failed: {exc}")
            with st.expander("Show full traceback"):
                import traceback
                st.code(traceback.format_exc(), language="python")
            st.stop()

        # --- Display results ---
        st.markdown("---")

        res_tab_sparams, res_tab_smith, res_tab_field, res_tab_time, res_tab_export = st.tabs(
            ["S-Parameters", "Smith Chart", "Field Snapshot", "Time Series", "Export"],
        )

        # S-Parameters
        with res_tab_sparams:
            if result.s_params is not None and result.freqs is not None:
                display_s_params(result.s_params, result.freqs)
            else:
                st.info("No S-parameter data. Add ports to compute S-parameters.")

        # Smith chart
        with res_tab_smith:
            if result.s_params is not None and result.freqs is not None:
                s11 = result.s_params[0, 0, :]
                display_smith_chart(s11, result.freqs)
            else:
                st.info("No S11 data available.")

        # Field snapshot
        with res_tab_field:
            col_fc, col_fa = st.columns(2)
            with col_fc:
                field_comp = st.selectbox(
                    "Field component",
                    ["ez", "ex", "ey", "hx", "hy", "hz"],
                    key="field_comp",
                )
            with col_fa:
                field_axis = st.selectbox(
                    "Slice axis",
                    ["z", "x", "y"],
                    key="field_axis",
                )
            grid = sim._build_grid()
            display_field_slice(result.state, grid, component=field_comp, axis=field_axis)

        # Time series
        with res_tab_time:
            ts = np.asarray(result.time_series)
            if ts.size > 0 and ts.shape[-1] > 0:
                display_time_series(ts, result.dt)
            else:
                st.info("No probe time-series data. Add probes to record fields.")

        # Export
        with res_tab_export:
            st.subheader("Export Data")

            # Touchstone download
            if result.s_params is not None and result.freqs is not None:
                n_p = result.s_params.shape[0]
                touchstone_bytes = export_touchstone(result.s_params, result.freqs)
                st.download_button(
                    f"Download .s{n_p}p (Touchstone)",
                    data=touchstone_bytes,
                    file_name=f"rfx_sim.s{n_p}p",
                    mime="application/octet-stream",
                    key="dl_touchstone",
                )

            # Field data download (numpy)
            if result.state is not None:
                import io as _io

                buf = _io.BytesIO()
                field_data = {}
                for comp in ("ex", "ey", "ez", "hx", "hy", "hz"):
                    arr = getattr(result.state, comp, None)
                    if arr is not None:
                        field_data[comp] = np.asarray(arr)
                np.savez_compressed(buf, **field_data)
                buf.seek(0)

                st.download_button(
                    "Download field data (.npz)",
                    data=buf.getvalue(),
                    file_name="rfx_fields.npz",
                    mime="application/octet-stream",
                    key="dl_fields",
                )

            # Time series CSV
            ts = np.asarray(result.time_series)
            if ts.size > 0 and ts.shape[-1] > 0:
                import io as _io

                if ts.ndim == 1:
                    ts = ts[:, np.newaxis]
                header = "time_step," + ",".join(f"probe_{i}" for i in range(ts.shape[1]))
                buf = _io.StringIO()
                buf.write(header + "\n")
                for row_idx in range(ts.shape[0]):
                    vals = ",".join(f"{ts[row_idx, c]:.8e}" for c in range(ts.shape[1]))
                    buf.write(f"{row_idx},{vals}\n")

                st.download_button(
                    "Download time series (.csv)",
                    data=buf.getvalue(),
                    file_name="rfx_time_series.csv",
                    mime="text/csv",
                    key="dl_csv",
                )

            if result.s_params is None and ts.size == 0:
                st.info("No data to export. Run a simulation with ports or probes first.")
