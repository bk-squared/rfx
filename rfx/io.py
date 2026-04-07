"""Touchstone (.sNp) S-parameter file I/O.

Reads and writes industry-standard Touchstone v1 format for interoperability
with ADS, CST, HFSS, and other RF tools.

Supports:
- .s1p (1-port), .s2p (2-port), .s4p (4-port), .snp (N-port)
- Data formats: RI (real/imaginary), MA (magnitude/angle), DB (dB/angle)
- Frequency units: Hz, kHz, MHz, GHz

Multi-port layout (Touchstone v1):
- 1-port and 2-port: all S-parameter data on one line per frequency
- 3+ ports: max 4 S-parameter pairs per line; first line starts with
  frequency, continuation lines are indented without a frequency column.
  Column-major order: S11, S21, ..., SN1, S12, ..., SNN.
"""

from __future__ import annotations

import numpy as np
from pathlib import Path

# Maximum number of S-parameter pairs per line for N>=3 ports (Touchstone v1).
_MAX_PAIRS_PER_LINE = 4


def _format_pair(s: complex, fmt: str) -> tuple[str, str]:
    """Format a single complex S-parameter into two string tokens."""
    if fmt == "RI":
        return f"{s.real:.9e}", f"{s.imag:.9e}"
    elif fmt == "MA":
        return f"{abs(s):.9e}", f"{np.degrees(np.angle(s)):.6f}"
    elif fmt == "DB":
        mag_db = 20 * np.log10(max(abs(s), 1e-15))
        return f"{mag_db:.6f}", f"{np.degrees(np.angle(s)):.6f}"
    else:
        raise ValueError(f"Unknown format: {fmt!r}")


def _parse_pair(v1: float, v2: float, fmt: str) -> complex:
    """Convert two raw Touchstone values into a complex S-parameter."""
    if fmt == "RI":
        return complex(v1, v2)
    elif fmt == "MA":
        return v1 * np.exp(1j * np.radians(v2))
    elif fmt == "DB":
        mag = 10 ** (v1 / 20)
        return mag * np.exp(1j * np.radians(v2))
    else:
        raise ValueError(f"Unknown format: {fmt!r}")


def write_touchstone(
    filepath: str | Path,
    s_params: np.ndarray,
    freqs: np.ndarray,
    z0: float = 50.0,
    freq_unit: str = "GHz",
    fmt: str = "RI",
    comments: list[str] | None = None,
) -> None:
    """Write S-parameters to a Touchstone v1 file.

    Parameters
    ----------
    filepath : str or Path
        Output file (extension determines port count: .s1p, .s2p, etc.)
    s_params : (n_ports, n_ports, n_freqs) complex array
    freqs : (n_freqs,) float in Hz
    z0 : reference impedance (ohm)
    freq_unit : "Hz", "kHz", "MHz", or "GHz"
    fmt : "RI" (real/imag), "MA" (mag/angle), or "DB" (dB/angle)
    comments : optional comment lines
    """
    s_params = np.asarray(s_params)
    freqs = np.asarray(freqs)
    n_ports = s_params.shape[0]
    n_freqs = s_params.shape[2]

    scale = {"Hz": 1.0, "kHz": 1e3, "MHz": 1e6, "GHz": 1e9}[freq_unit]

    with open(filepath, "w") as f:
        # Comments
        f.write("! rfx Touchstone export\n")
        f.write(f"! {n_ports}-port, {n_freqs} frequencies\n")
        if comments:
            for c in comments:
                f.write(f"! {c}\n")

        # Option line
        f.write(f"# {freq_unit} S {fmt} R {z0:.1f}\n")

        # Data — layout depends on port count
        for fi in range(n_freqs):
            freq_scaled = freqs[fi] / scale

            # Collect all S-parameter pairs in column-major order:
            # S11, S21, ..., SN1, S12, S22, ..., SN2, ...
            pairs: list[tuple[str, str]] = []
            for j in range(n_ports):
                for i in range(n_ports):
                    pairs.append(_format_pair(s_params[i, j, fi], fmt))

            if n_ports <= 2:
                # 1-port and 2-port: everything on one line
                tokens = [f"{freq_scaled:.9e}"]
                for p1, p2 in pairs:
                    tokens.extend([p1, p2])
                f.write(" ".join(tokens) + "\n")
            else:
                # 3+ ports: first line has freq + up to 4 pairs,
                # continuation lines have up to 4 pairs each.
                idx = 0
                first_line_tokens = [f"{freq_scaled:.9e}"]
                for _ in range(min(_MAX_PAIRS_PER_LINE, len(pairs))):
                    p1, p2 = pairs[idx]
                    first_line_tokens.extend([p1, p2])
                    idx += 1
                f.write(" ".join(first_line_tokens) + "\n")

                # Continuation lines
                while idx < len(pairs):
                    chunk_end = min(idx + _MAX_PAIRS_PER_LINE, len(pairs))
                    cont_tokens: list[str] = []
                    while idx < chunk_end:
                        p1, p2 = pairs[idx]
                        cont_tokens.extend([p1, p2])
                        idx += 1
                    # Indent continuation lines to distinguish from freq lines
                    f.write("  " + " ".join(cont_tokens) + "\n")


def read_touchstone(filepath: str | Path) -> tuple[np.ndarray, np.ndarray, float]:
    """Read S-parameters from a Touchstone v1 file.

    Handles both single-line (1- and 2-port) and multi-line (3+ port) data
    blocks.  Inline comments (``! ...``) after data values are stripped.

    Parameters
    ----------
    filepath : str or Path

    Returns
    -------
    s_params : (n_ports, n_ports, n_freqs) complex array
    freqs : (n_freqs,) float in Hz
    z0 : float reference impedance
    """
    filepath = Path(filepath)

    # Infer port count from extension
    ext = filepath.suffix.lower()
    if ext.startswith(".s") and ext.endswith("p"):
        n_ports = int(ext[2:-1])
    else:
        raise ValueError(f"Cannot determine port count from extension: {ext}")

    freq_scale = 1e9  # default GHz
    fmt = "RI"
    z0 = 50.0

    raw_data_lines: list[str] = []

    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("!"):
                continue
            if line.startswith("#"):
                # Parse option line
                tokens = line[1:].split()
                i = 0
                while i < len(tokens):
                    t = tokens[i].upper()
                    if t in ("HZ", "KHZ", "MHZ", "GHZ"):
                        freq_scale = {"HZ": 1.0, "KHZ": 1e3, "MHZ": 1e6, "GHZ": 1e9}[t]
                    elif t in ("RI", "MA", "DB"):
                        fmt = t
                    elif t == "R" and i + 1 < len(tokens):
                        z0 = float(tokens[i + 1])
                        i += 1
                    i += 1
                continue

            # Strip inline comments: "1.0 0.5 0.3 ! my comment"
            if "!" in line:
                line = line[:line.index("!")]
            line = line.strip()
            if line:
                raw_data_lines.append(line)

    # Number of value pairs expected per frequency point
    n_pairs = n_ports * n_ports  # one complex pair per S-parameter
    n_values_expected = 1 + 2 * n_pairs  # freq + 2 floats per pair

    # Merge raw lines into logical frequency-point blocks.
    # For 1- and 2-port files every raw line is a complete block.
    # For 3+ port files, continuation lines lack a frequency column
    # and must be merged with the preceding frequency line.
    all_values: list[float] = []
    for line in raw_data_lines:
        vals = [float(x) for x in line.split()]
        all_values.extend(vals)

    # Each frequency point contributes exactly n_values_expected floats
    if len(all_values) % n_values_expected != 0:
        raise ValueError(
            f"Data length {len(all_values)} is not a multiple of "
            f"expected {n_values_expected} values per frequency point "
            f"(n_ports={n_ports})"
        )

    n_freqs = len(all_values) // n_values_expected

    # Parse into structured arrays
    freqs_list: list[float] = []
    s_data: list[list[complex]] = []

    offset = 0
    for _ in range(n_freqs):
        freq_val = all_values[offset] * freq_scale
        freqs_list.append(freq_val)
        offset += 1

        row: list[complex] = []
        for _ in range(n_pairs):
            v1 = all_values[offset]
            v2 = all_values[offset + 1]
            row.append(_parse_pair(v1, v2, fmt))
            offset += 2
        s_data.append(row)

    freqs = np.array(freqs_list)

    # Touchstone stores network data column-major:
    # S11, S21, ..., SN1, S12, S22, ..., SN2, ...
    s_params = np.zeros((n_ports, n_ports, n_freqs), dtype=np.complex128)
    for fi in range(n_freqs):
        idx = 0
        for j in range(n_ports):
            for i in range(n_ports):
                s_params[i, j, fi] = s_data[fi][idx]
                idx += 1

    return s_params, freqs, z0


# =========================================================================
# Result export (HDF5)
# =========================================================================

def save_optimization_result(path, result, metadata=None):
    """Save OptimizeResult or TopologyResult to HDF5.

    Parameters
    ----------
    path : str or Path
    result : OptimizeResult or TopologyResult
    metadata : dict or None
        Extra metadata (grid shape, freq_max, etc.) stored as HDF5 attrs.
    """
    import h5py

    path = Path(path)
    with h5py.File(path, "w") as f:
        f.create_dataset("eps_design", data=np.asarray(result.eps_design))

        if hasattr(result, "loss_history"):
            f.create_dataset("loss_history", data=np.array(result.loss_history))
        if hasattr(result, "history"):
            f.create_dataset("loss_history", data=np.array(result.history))

        if hasattr(result, "latent") and result.latent is not None:
            f.create_dataset("latent", data=np.asarray(result.latent))
        if hasattr(result, "density") and result.density is not None:
            f.create_dataset("density", data=np.asarray(result.density))
        if hasattr(result, "density_projected") and result.density_projected is not None:
            f.create_dataset("density_projected", data=np.asarray(result.density_projected))
        if hasattr(result, "beta_history"):
            f.create_dataset("beta_history", data=np.array(result.beta_history))
        if hasattr(result, "pec_occupancy_design") and result.pec_occupancy_design is not None:
            f.create_dataset("pec_occupancy", data=np.asarray(result.pec_occupancy_design))

        if metadata:
            for k, v in metadata.items():
                f.attrs[k] = v


def load_optimization_result(path):
    """Load optimization result from HDF5.

    Returns
    -------
    dict with keys: eps_design, loss_history, latent, density, etc.
    """
    import h5py

    path = Path(path)
    data = {}
    with h5py.File(path, "r") as f:
        for key in f.keys():
            data[key] = np.array(f[key])
        data["metadata"] = dict(f.attrs)
    return data


def save_far_field(path, ff_result, metadata=None):
    """Save FarFieldResult to HDF5.

    Parameters
    ----------
    path : str or Path
    ff_result : FarFieldResult
    metadata : dict or None
    """
    import h5py

    path = Path(path)
    with h5py.File(path, "w") as f:
        f.create_dataset("E_theta", data=np.asarray(ff_result.E_theta))
        f.create_dataset("E_phi", data=np.asarray(ff_result.E_phi))
        f.create_dataset("theta", data=np.asarray(ff_result.theta))
        f.create_dataset("phi", data=np.asarray(ff_result.phi))
        f.create_dataset("freqs", data=np.asarray(ff_result.freqs))

        # Derived: directivity
        E_th = np.asarray(ff_result.E_theta)
        E_ph = np.asarray(ff_result.E_phi)
        power = np.abs(E_th) ** 2 + np.abs(E_ph) ** 2
        f.create_dataset("power_pattern", data=power)

        if metadata:
            for k, v in metadata.items():
                f.attrs[k] = v


def export_radiation_pattern(path, ff_result, freq_idx=0):
    """Export radiation pattern as CSV for measurement comparison.

    Columns: theta_deg, phi_deg, E_theta_mag, E_theta_phase_deg,
             E_phi_mag, E_phi_phase_deg, gain_dBi

    Parameters
    ----------
    path : str or Path
    ff_result : FarFieldResult
    freq_idx : int
        Frequency index to export.
    """
    path = Path(path)
    E_th = np.asarray(ff_result.E_theta[freq_idx])  # (n_theta, n_phi)
    E_ph = np.asarray(ff_result.E_phi[freq_idx])
    theta = np.asarray(ff_result.theta)
    phi = np.asarray(ff_result.phi)

    TH, PH = np.meshgrid(theta, phi, indexing="ij")
    power = np.abs(E_th) ** 2 + np.abs(E_ph) ** 2
    max_power = np.max(power)
    gain_dbi = 10 * np.log10(power / max(max_power, 1e-30) + 1e-30)

    rows = []
    for i in range(len(theta)):
        for j in range(len(phi)):
            rows.append([
                np.degrees(theta[i]), np.degrees(phi[j]),
                np.abs(E_th[i, j]), np.degrees(np.angle(E_th[i, j])),
                np.abs(E_ph[i, j]), np.degrees(np.angle(E_ph[i, j])),
                gain_dbi[i, j],
            ])

    header = "theta_deg,phi_deg,E_theta_mag,E_theta_phase_deg,E_phi_mag,E_phi_phase_deg,gain_dBi"
    np.savetxt(path, rows, delimiter=",", header=header, comments="", fmt="%.6e")


# =========================================================================
# Geometry export (JSON)
# =========================================================================

def export_geometry_json(path, sim):
    """Export simulation geometry as JSON for cross-validation or dataset indexing.

    Captures materials, geometry entries, sources, probes, and grid config
    in a tool-agnostic format.

    Parameters
    ----------
    path : str or Path
    sim : Simulation
    """
    import json

    geo = {
        "freq_max": sim._freq_max,
        "domain": list(sim._domain),
        "boundary": sim._boundary,
        "dx": sim._dx,
        "mode": sim._mode,
        "cpml_layers": sim._cpml_layers,
        "materials": {
            name: {"eps_r": float(m.eps_r), "sigma": float(m.sigma), "mu_r": float(m.mu_r)}
            for name, m in sim._materials.items()
        },
        "geometry": [
            {
                "material": e.material_name,
                "type": type(e.shape).__name__,
                "bbox": [list(e.shape.bounding_box()[0]),
                         list(e.shape.bounding_box()[1])]
                if hasattr(e.shape, "bounding_box") else None,
            }
            for e in sim._geometry
        ],
        "sources": [
            {"position": list(p.position), "component": p.component}
            for p in sim._ports if p.impedance == 0
        ],
        "probes": [
            {"position": list(p.position), "component": p.component}
            for p in sim._probes
        ],
    }

    path = Path(path)
    with open(path, "w") as f:
        json.dump(geo, f, indent=2, default=str)


# =========================================================================
# Experiment report
# =========================================================================

def save_experiment_report(path, sim, result, extra=None):
    """Save standardized experiment metadata as JSON.

    Auto-collects: sim config, grid shape, timing, loss history.

    Parameters
    ----------
    path : str or Path
    sim : Simulation
    result : Result, OptimizeResult, or TopologyResult
    extra : dict or None
        Additional metadata (GPU info, notes, etc.)
    """
    import json
    import time

    grid = None
    if hasattr(result, "grid") and result.grid is not None:
        grid = result.grid

    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "simulation": {
            "freq_max": sim._freq_max,
            "domain": list(sim._domain),
            "boundary": sim._boundary,
            "dx": sim._dx,
            "mode": sim._mode,
            "solver": getattr(sim, "_solver", "yee"),
        },
        "grid": {
            "shape": list(grid.shape) if grid else None,
            "dx": float(grid.dx) if grid else None,
            "dt": float(grid.dt) if grid else None,
        } if grid else None,
        "result_type": type(result).__name__,
    }

    # Loss history from optimization results
    if hasattr(result, "loss_history"):
        report["loss_history"] = [float(x) for x in result.loss_history]
    elif hasattr(result, "history"):
        report["loss_history"] = [float(x) for x in result.history]

    # Time series stats
    if hasattr(result, "time_series") and result.time_series is not None:
        ts = np.asarray(result.time_series)
        report["time_series"] = {
            "shape": list(ts.shape),
            "peak": float(np.max(np.abs(ts))),
        }

    if extra:
        report["extra"] = extra

    path = Path(path)
    with open(path, "w") as f:
        json.dump(report, f, indent=2, default=str)


# =========================================================================
# Surrogate/ML training data export
# =========================================================================

def save_simulation_dataset(path, sim, result, *, include_fields=False):
    """Save simulation input-output pair for surrogate model training.

    Creates an HDF5 file with:
    - Input: material distribution (eps_r, sigma), geometry config
    - Output: time series, S-parameters, resonances
    - Optionally: final field snapshot (E, H)

    Parameters
    ----------
    path : str or Path
    sim : Simulation
    result : Result
    include_fields : bool
        If True, save full 3D E/H field arrays (large).
    """
    import h5py

    path = Path(path)
    with h5py.File(path, "w") as f:
        # Input group: material distribution
        inp = f.create_group("input")
        inp.attrs["freq_max"] = sim._freq_max
        inp.attrs["domain"] = sim._domain
        inp.attrs["boundary"] = sim._boundary
        if sim._dx is not None:
            inp.attrs["dx"] = sim._dx

        # Rasterized materials from grid
        if hasattr(result, "grid") and result.grid is not None:
            grid = result.grid
            inp.attrs["grid_shape"] = grid.shape
            inp.attrs["dt"] = float(grid.dt)

        # Output group
        out = f.create_group("output")

        # Time series
        if hasattr(result, "time_series") and result.time_series is not None:
            ts = np.asarray(result.time_series)
            out.create_dataset("time_series", data=ts, compression="gzip")

        # S-parameters
        if hasattr(result, "s_params") and result.s_params is not None:
            out.create_dataset("s_params", data=np.asarray(result.s_params))
        if hasattr(result, "freqs") and result.freqs is not None:
            out.create_dataset("freqs", data=np.asarray(result.freqs))

        # Field snapshots (optional, large)
        if include_fields and hasattr(result, "state") and result.state is not None:
            fields = f.create_group("fields")
            for comp in ("ex", "ey", "ez", "hx", "hy", "hz"):
                arr = getattr(result.state, comp, None)
                if arr is not None:
                    fields.create_dataset(comp, data=np.asarray(arr),
                                          compression="gzip")


def save_optimization_trajectory(path, history_callback_data):
    """Save full optimization trajectory for meta-learning / warm-start.

    Parameters
    ----------
    path : str or Path
    history_callback_data : list of dict
        Each entry: {"iter": int, "loss": float, "eps_design": ndarray,
                     "s_params": ndarray (optional)}
    """
    import h5py

    path = Path(path)
    n = len(history_callback_data)
    if n == 0:
        return

    with h5py.File(path, "w") as f:
        f.attrs["n_iterations"] = n
        losses = [d["loss"] for d in history_callback_data]
        f.create_dataset("losses", data=np.array(losses))

        # Save designs at each iteration (compressed)
        for i, d in enumerate(history_callback_data):
            g = f.create_group(f"iter_{i:04d}")
            g.attrs["loss"] = d["loss"]
            g.create_dataset("eps_design", data=np.asarray(d["eps_design"]),
                             compression="gzip")
            if "s_params" in d and d["s_params"] is not None:
                g.create_dataset("s_params", data=np.asarray(d["s_params"]))
            if "density" in d and d["density"] is not None:
                g.create_dataset("density", data=np.asarray(d["density"]))
