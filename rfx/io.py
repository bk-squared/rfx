"""Touchstone (.sNp) S-parameter file I/O.

Reads and writes industry-standard Touchstone format for interoperability
with ADS, CST, HFSS, and other RF tools.

Supports Touchstone v1 format with RI (real/imaginary), MA (magnitude/angle),
and DB (dB/angle) data formats.
"""

from __future__ import annotations

import numpy as np
from pathlib import Path


def write_touchstone(
    filepath: str | Path,
    s_params: np.ndarray,
    freqs: np.ndarray,
    z0: float = 50.0,
    freq_unit: str = "GHz",
    fmt: str = "RI",
    comments: list[str] | None = None,
) -> None:
    """Write S-parameters to a Touchstone file.

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
        f.write(f"! rfx Touchstone export\n")
        f.write(f"! {n_ports}-port, {n_freqs} frequencies\n")
        if comments:
            for c in comments:
                f.write(f"! {c}\n")

        # Option line
        f.write(f"# {freq_unit} S {fmt} R {z0:.1f}\n")

        # Data
        for fi in range(n_freqs):
            freq_scaled = freqs[fi] / scale
            parts = [f"{freq_scaled:.9e}"]

            # Touchstone stores network data column-major:
            # S11, S21, ..., SN1, S12, S22, ..., SN2, ...
            for j in range(n_ports):
                for i in range(n_ports):
                    s = s_params[i, j, fi]
                    if fmt == "RI":
                        parts.append(f"{s.real:.9e}")
                        parts.append(f"{s.imag:.9e}")
                    elif fmt == "MA":
                        parts.append(f"{abs(s):.9e}")
                        parts.append(f"{np.degrees(np.angle(s)):.6f}")
                    elif fmt == "DB":
                        mag_db = 20 * np.log10(max(abs(s), 1e-15))
                        parts.append(f"{mag_db:.6f}")
                        parts.append(f"{np.degrees(np.angle(s)):.6f}")

            f.write(" ".join(parts) + "\n")


def read_touchstone(filepath: str | Path) -> tuple[np.ndarray, np.ndarray, float]:
    """Read S-parameters from a Touchstone file.

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

    data_lines = []

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

            data_lines.append(line)

    # Parse data
    freqs_list = []
    s_data = []
    for line in data_lines:
        vals = [float(x) for x in line.split()]
        freqs_list.append(vals[0] * freq_scale)
        pairs = vals[1:]
        row = []
        for k in range(0, len(pairs), 2):
            v1, v2 = pairs[k], pairs[k + 1]
            if fmt == "RI":
                row.append(complex(v1, v2))
            elif fmt == "MA":
                row.append(v1 * np.exp(1j * np.radians(v2)))
            elif fmt == "DB":
                mag = 10 ** (v1 / 20)
                row.append(mag * np.exp(1j * np.radians(v2)))
        s_data.append(row)

    freqs = np.array(freqs_list)
    n_freqs = len(freqs)

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
