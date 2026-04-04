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
