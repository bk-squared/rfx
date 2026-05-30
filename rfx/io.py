"""Touchstone (.sNp) S-parameter file I/O.

Reads and writes the legacy rfx Touchstone v1-compatible format plus a bounded
metadata-aware Touchstone 2.0 S-parameter subset for interoperability with ADS,
CST, HFSS, scikit-rf-style workflows, and other RF tools.

Supports:
- .s1p (1-port), .s2p (2-port), .s4p (4-port), .sNp (N-port by extension)
- Data formats: RI (real/imaginary), MA (magnitude/angle), DB (dB/angle)
- Frequency units: Hz, kHz, MHz, GHz
- Touchstone 2.0 metadata subset: [Version], [Number of Ports],
  [Number of Frequencies], [Reference], [Matrix Format] Full,
  [Two-Port Data Order], [Begin Information]/[End Information],
  [Network Data], [End]

Legacy rfx multi-port layout:
- 1-port and 2-port: all S-parameter data on one line per frequency
- 3+ ports: max 4 S-parameter pairs per line; first line starts with
  frequency, continuation lines are indented without a frequency column.
  Column-major order: S11, S21, ..., SN1, S12, ..., SNN.

Standard Touchstone layout is available with ``layout="standard"`` or
``version="2.0"`` and uses row-wise 3+ port full matrices.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Maximum number of S-parameter pairs per line for N>=3 ports (Touchstone v1).
_MAX_PAIRS_PER_LINE = 4


@dataclass(frozen=True)
class TouchstoneData:
    """Metadata-aware Touchstone read result.

    The legacy :func:`read_touchstone` API intentionally returns only
    ``(s_params, freqs, z0)``.  Use this dataclass through
    :func:`read_touchstone_full` when Touchstone 2.0 metadata such as per-port
    reference impedances or explicit matrix layout matters.
    """

    s_params: np.ndarray
    freqs: np.ndarray
    z0: float | np.ndarray
    reference: np.ndarray
    version: str
    parameter: str
    fmt: str
    freq_unit: str
    layout: str
    matrix_format: str
    two_port_order: str
    comments: tuple[str, ...] = ()
    information: tuple[str, ...] = ()


def _parse_touchstone_float(token: str) -> float:
    """Parse a Touchstone numeric token.

    Some RF tools emit Fortran-style ``D`` exponents (for example
    ``1.0D+09``).  Python's ``float`` does not accept those, but Touchstone
    numeric data should be tolerant of both ``E`` and ``D`` exponent markers.
    """
    return float(token.replace("D", "E").replace("d", "e"))


def _parse_touchstone_int(token: str) -> int:
    """Parse an integer keyword value, accepting exponent notation."""
    value = _parse_touchstone_float(token)
    integer = int(value)
    if integer != value:
        raise ValueError(f"Expected integer Touchstone value, got {token!r}")
    return integer


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


def _reference_vector(z0: float | np.ndarray, port_z0: np.ndarray | None,
                      n_ports: int) -> np.ndarray:
    """Return one reference impedance per port."""
    if port_z0 is None:
        refs = np.full(n_ports, float(z0), dtype=float)
    else:
        refs = np.asarray(port_z0, dtype=float).reshape(-1)
        if refs.shape != (n_ports,):
            raise ValueError(
                f"port_z0 must contain one value per port "
                f"(expected {n_ports}, got {refs.size})"
            )
    if not np.all(np.isfinite(refs)):
        raise ValueError("reference impedances must be finite")
    return refs


def _uniform_reference_or_none(reference: np.ndarray) -> float | None:
    """Return scalar reference if all ports share one value."""
    refs = np.asarray(reference, dtype=float).reshape(-1)
    if refs.size == 0:
        return None
    if np.allclose(refs, refs[0], rtol=0.0, atol=0.0):
        return float(refs[0])
    return None


def _pair_indices(n_ports: int, *, layout: str,
                  two_port_order: str) -> list[tuple[int, int]]:
    """Return ``(receiver, driven)`` order for one frequency block."""
    layout = layout.lower()
    order = two_port_order.upper()

    if n_ports == 1:
        return [(0, 0)]

    if n_ports == 2:
        if order in ("21_12", "S21_S12"):
            return [(0, 0), (1, 0), (0, 1), (1, 1)]
        if order in ("12_21", "S12_S21"):
            return [(0, 0), (0, 1), (1, 0), (1, 1)]
        raise ValueError(
            "two_port_order must be '21_12' or '12_21' for 2-port data"
        )

    if layout in ("legacy-rfx", "legacy", "column-major"):
        # Historical rfx layout: S11, S21, ..., SN1, S12, ...
        return [(i, j) for j in range(n_ports) for i in range(n_ports)]
    if layout in ("standard", "touchstone", "row-major"):
        # Touchstone 2.0 full matrix layout for 3+ ports: rows of S.
        return [(i, j) for i in range(n_ports) for j in range(n_ports)]
    raise ValueError("layout must be 'legacy-rfx' or 'standard'")


def _format_frequency_block(
    s_params: np.ndarray,
    fi: int,
    fmt: str,
    *,
    layout: str,
    two_port_order: str,
) -> list[tuple[str, str]]:
    n_ports = s_params.shape[0]
    return [
        _format_pair(s_params[i, j, fi], fmt)
        for i, j in _pair_indices(
            n_ports, layout=layout, two_port_order=two_port_order
        )
    ]


def _infer_touchstone_port_count(filepath: str | Path) -> int | None:
    """Infer numeric port count from a Touchstone suffix."""
    ext = Path(filepath).suffix.lower()
    if not (ext.startswith(".s") and ext.endswith("p")):
        raise ValueError(f"Cannot determine port count from extension: {ext}")
    token = ext[2:-1]
    return int(token) if token.isdigit() else None


def _information_lines(
    information: dict[str, object] | list[str] | tuple[str, ...] | None,
) -> list[str]:
    """Normalize optional Touchstone 2.0 information-block content."""
    if information is None:
        return []
    if isinstance(information, dict):
        return [
            f"{key} {information[key]}"
            for key in sorted(information)
        ]
    return [str(line) for line in information]


def write_touchstone(
    filepath: str | Path,
    s_params: np.ndarray,
    freqs: np.ndarray,
    z0: float = 50.0,
    freq_unit: str = "GHz",
    fmt: str = "RI",
    comments: list[str] | None = None,
    *,
    version: str = "1.0",
    layout: str | None = None,
    port_z0: np.ndarray | None = None,
    matrix_format: str = "Full",
    two_port_order: str = "21_12",
    information: dict[str, object] | list[str] | tuple[str, ...] | None = None,
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
    version : "1.0" or "2.0"
        ``"1.0"`` preserves the historical rfx writer shape.  ``"2.0"``
        writes Touchstone 2.0 metadata keywords for the supported S-parameter
        subset.
    layout : "legacy-rfx", "standard", or None
        Multi-port order.  ``None`` means historical legacy layout for
        Touchstone 1.0 output and standard layout for Touchstone 2.0 output.
        The legacy rfx layout is column-major for all port counts.  The
        standard layout is row-major for 3+ ports.  Two-port standard order is
        controlled by ``two_port_order``.
    port_z0 : array-like or None
        Optional per-port reference impedances.  Requires ``version="2.0"``
        when non-uniform.
    matrix_format : str
        Touchstone 2.0 matrix format.  Only ``"Full"`` is supported.
    two_port_order : "21_12" or "12_21"
        Explicit 2-port data order for standard Touchstone 2.0 files.
    information : dict, list, tuple, or None
        Optional Touchstone 2.0 information block.  Dicts are written as
        sorted ``"key value"`` lines; sequences are written verbatim.
    """
    s_params = np.asarray(s_params)
    freqs = np.asarray(freqs)
    if s_params.ndim != 3 or s_params.shape[0] != s_params.shape[1]:
        raise ValueError("s_params must have shape (n_ports, n_ports, n_freqs)")
    n_ports = s_params.shape[0]
    n_freqs = s_params.shape[2]
    if freqs.shape != (n_freqs,):
        raise ValueError("freqs must have shape (n_freqs,)")
    if not np.all(np.isfinite(s_params)):
        raise ValueError("s_params must contain only finite values")
    if not np.all(np.isfinite(freqs)):
        raise ValueError("freqs must contain only finite values")

    version = str(version)
    if version not in ("1.0", "2.0"):
        raise ValueError("version must be '1.0' or '2.0'")
    ext_n_ports = _infer_touchstone_port_count(filepath)
    if ext_n_ports is not None and ext_n_ports != n_ports:
        raise ValueError(
            f"file extension declares {ext_n_ports} ports, "
            f"but s_params has {n_ports}"
        )
    if ext_n_ports is None and version != "2.0":
        raise ValueError(
            "non-numeric .sNp extensions require version='2.0' "
            "with [Number of Ports] metadata"
        )
    if layout is None:
        layout = "standard" if version == "2.0" else "legacy-rfx"
    layout = layout.lower()
    fmt = fmt.upper()
    matrix_format = matrix_format.capitalize()
    two_port_order = two_port_order.upper()
    if matrix_format != "Full":
        raise ValueError("Only Touchstone matrix_format='Full' is supported")
    info_lines = _information_lines(information)
    if info_lines and version != "2.0":
        raise ValueError("Touchstone information blocks require version='2.0'")

    refs = _reference_vector(z0, port_z0, n_ports)
    uniform_z0 = _uniform_reference_or_none(refs)
    if version != "2.0" and uniform_z0 is None:
        raise ValueError("non-uniform port_z0 requires version='2.0'")

    try:
        scale = {"Hz": 1.0, "kHz": 1e3, "MHz": 1e6, "GHz": 1e9}[freq_unit]
    except KeyError as exc:
        raise ValueError("freq_unit must be 'Hz', 'kHz', 'MHz', or 'GHz'") from exc
    option_z0 = uniform_z0 if uniform_z0 is not None else float(refs[0])

    with open(filepath, "w") as f:
        # Comments
        f.write("! rfx Touchstone export\n")
        f.write(f"! {n_ports}-port, {n_freqs} frequencies\n")
        if comments:
            for c in comments:
                f.write(f"! {c}\n")

        if version == "2.0":
            f.write("[Version] 2.0\n")

        # Option line
        f.write(f"# {freq_unit} S {fmt} R {option_z0:.12g}\n")

        if version == "2.0":
            f.write(f"[Number of Ports] {n_ports}\n")
            f.write(f"[Number of Frequencies] {n_freqs}\n")
            f.write("[Reference] " + " ".join(f"{v:.12g}" for v in refs) + "\n")
            f.write(f"[Matrix Format] {matrix_format}\n")
            if n_ports == 2:
                f.write(f"[Two-Port Data Order] {two_port_order}\n")
            if info_lines:
                f.write("[Begin Information]\n")
                for line in info_lines:
                    f.write(f"{line}\n")
                f.write("[End Information]\n")
            f.write("[Network Data]\n")

        # Data — layout depends on port count
        for fi in range(n_freqs):
            freq_scaled = freqs[fi] / scale

            pairs = _format_frequency_block(
                s_params,
                fi,
                fmt,
                layout=layout,
                two_port_order=two_port_order,
            )

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

        if version == "2.0":
            f.write("[End]\n")


def read_touchstone(filepath: str | Path) -> tuple[np.ndarray, np.ndarray, float]:
    """Read S-parameters from a Touchstone file.

    This legacy compatibility API returns a scalar reference impedance.  If a
    Touchstone 2.0 file carries non-uniform per-port references, use
    :func:`read_touchstone_full` instead.

    Parameters
    ----------
    filepath : str or Path

    Returns
    -------
    s_params : (n_ports, n_ports, n_freqs) complex array
    freqs : (n_freqs,) float in Hz
    z0 : float reference impedance
    """
    data = read_touchstone_full(filepath)
    scalar_z0 = _uniform_reference_or_none(data.reference)
    if scalar_z0 is None:
        raise ValueError(
            "Touchstone file has non-uniform per-port [Reference] values; "
            "use read_touchstone_full() to preserve reference metadata"
        )
    return data.s_params, data.freqs, scalar_z0


def read_touchstone_full(filepath: str | Path, *,
                         layout: str = "auto") -> TouchstoneData:
    """Read S-parameters and Touchstone metadata.

    Supported subset:
    - S-parameters only
    - RI, MA, and DB numeric formats
    - Touchstone 1-style data blocks
    - Touchstone 2.0 metadata keywords used by rfx interop
    - Touchstone 2.0 information blocks preserved as raw lines
    - Full matrix data only

    For backwards compatibility, Touchstone 1-style files default to the
    historical rfx multi-port layout.  Touchstone 2.0 files default to the
    standard full-matrix row-wise layout for 3+ ports.
    """
    filepath = Path(filepath)

    # Infer port count from extension
    ext_n_ports = _infer_touchstone_port_count(filepath)
    n_ports = ext_n_ports

    freq_unit = "GHz"
    freq_scale = 1e9  # default GHz
    # Preserve old no-option-line rfx behavior as RI, but Touchstone option
    # lines default to MA when the format token is omitted.
    fmt = "RI"
    z0 = 50.0
    version = "1.0"
    parameter = "S"
    matrix_format = "Full"
    two_port_order = "21_12"
    reference_values: list[float] | None = None
    declared_n_freqs: int | None = None
    comments: list[str] = []
    information: list[str] = []
    has_v2_keywords = False

    raw_data_lines: list[str] = []
    in_network_data = False
    in_reference = False
    in_information = False

    with open(filepath) as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            if in_information:
                if stripped.lower().startswith("[end information]"):
                    in_information = False
                else:
                    information.append(stripped)
                continue
            if stripped.startswith("!"):
                comments.append(stripped[1:].strip())
                continue
            if stripped.startswith("["):
                has_v2_keywords = True
                end = stripped.find("]")
                if end < 0:
                    raise ValueError(f"Malformed Touchstone keyword: {stripped}")
                key = stripped[1:end].strip().lower()
                rest = stripped[end + 1:].strip()
                if "!" in rest:
                    rest = rest[:rest.index("!")].strip()

                in_network_data = False
                in_reference = False

                if key == "version":
                    version = (rest.split()[0] if rest else version)
                    if version not in ("1.0", "2.0"):
                        raise ValueError(
                            f"Unsupported Touchstone [Version] {version!r}; "
                            "only 1.0/2.0 S-parameter files are supported"
                        )
                elif key == "number of ports":
                    if rest:
                        declared_n_ports = _parse_touchstone_int(rest.split()[0])
                        if ext_n_ports is not None and declared_n_ports != ext_n_ports:
                            raise ValueError(
                                "[Number of Ports] does not match file extension "
                                f"({declared_n_ports} != {ext_n_ports})"
                            )
                        n_ports = declared_n_ports
                elif key == "number of frequencies":
                    if rest:
                        declared_n_freqs = _parse_touchstone_int(rest.split()[0])
                elif key == "reference":
                    if n_ports is None:
                        raise ValueError(
                            "[Number of Ports] is required before [Reference] "
                            "when the extension does not encode the port count"
                        )
                    vals = [_parse_touchstone_float(x) for x in rest.split()] if rest else []
                    reference_values = vals
                    in_reference = rest == ""
                elif key == "matrix format":
                    matrix_format = (rest.split()[0] if rest else "Full").capitalize()
                    if matrix_format != "Full":
                        raise ValueError("Only Touchstone [Matrix Format] Full is supported")
                elif key == "two-port data order":
                    two_port_order = rest.upper() if rest else two_port_order
                elif key == "begin information":
                    in_information = True
                elif key == "end information":
                    raise ValueError("[End Information] without [Begin Information]")
                elif key == "network data":
                    if n_ports is None:
                        raise ValueError(
                            "Cannot determine port count; use an .sNp extension "
                            "with N digits or provide [Number of Ports] before "
                            "[Network Data]"
                        )
                    in_network_data = True
                elif key == "end":
                    break
                else:
                    raise ValueError(f"Unsupported Touchstone keyword [{key}]")
                continue

            # Strip inline comments: "1.0 0.5 0.3 ! my comment"
            line = stripped
            if "!" in line:
                line = line[:line.index("!")]
            line = line.strip()
            if not line:
                continue

            if in_reference:
                vals = [_parse_touchstone_float(x) for x in line.split()]
                reference_values = (reference_values or []) + vals
                if n_ports is None:
                    raise ValueError(
                        "[Number of Ports] is required before multi-line [Reference]"
                    )
                if len(reference_values) >= n_ports:
                    in_reference = False
                continue

            if line.startswith("#"):
                # Parse option line.  Touchstone option-line defaults are:
                # GHz, S, MA, R 50.  Existing no-option-line files keep the
                # historical RI fallback initialized above.
                freq_unit = "GHz"
                freq_scale = 1e9
                parameter = "S"
                fmt = "MA"
                z0 = 50.0
                tokens = line[1:].split()
                i = 0
                while i < len(tokens):
                    t = tokens[i].upper()
                    if t in ("HZ", "KHZ", "MHZ", "GHZ"):
                        freq_unit = {"HZ": "Hz", "KHZ": "kHz", "MHZ": "MHz", "GHZ": "GHz"}[t]
                        freq_scale = {"HZ": 1.0, "KHZ": 1e3, "MHZ": 1e6, "GHZ": 1e9}[t]
                    elif t in ("S", "Y", "Z", "G", "H"):
                        parameter = t
                        if parameter != "S":
                            raise ValueError("Only Touchstone S-parameter files are supported")
                    elif t in ("RI", "MA", "DB"):
                        fmt = t
                    elif t == "R" and i + 1 < len(tokens):
                        z0 = _parse_touchstone_float(tokens[i + 1])
                        i += 1
                    i += 1
                continue

            if in_network_data or not has_v2_keywords:
                raw_data_lines.append(line)

    if n_ports is None:
        raise ValueError(
            "Cannot determine port count; use an .sNp extension with N digits "
            "or provide [Number of Ports] metadata"
        )

    # Number of value pairs expected per frequency point
    n_pairs = n_ports * n_ports  # one complex pair per S-parameter
    n_values_expected = 1 + 2 * n_pairs  # freq + 2 floats per pair

    # Merge raw lines into logical frequency-point blocks.
    # For 1- and 2-port files every raw line is a complete block.
    # For 3+ port files, continuation lines lack a frequency column
    # and must be merged with the preceding frequency line.
    all_values: list[float] = []
    for line in raw_data_lines:
        vals = [_parse_touchstone_float(x) for x in line.split()]
        all_values.extend(vals)

    # Each frequency point contributes exactly n_values_expected floats
    if len(all_values) % n_values_expected != 0:
        raise ValueError(
            f"Data length {len(all_values)} is not a multiple of "
            f"expected {n_values_expected} values per frequency point "
            f"(n_ports={n_ports})"
        )

    n_freqs = len(all_values) // n_values_expected
    if declared_n_freqs is not None and n_freqs != declared_n_freqs:
        raise ValueError(
            "[Number of Frequencies] declares "
            f"{declared_n_freqs} points, but parsed {n_freqs}"
        )

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

    if layout == "auto":
        resolved_layout = "standard" if has_v2_keywords else "legacy-rfx"
    else:
        resolved_layout = layout.lower()

    if reference_values is None:
        reference = np.full(n_ports, z0, dtype=float)
    else:
        reference = np.asarray(reference_values, dtype=float).reshape(-1)
        if reference.shape != (n_ports,):
            raise ValueError(
                f"[Reference] must contain {n_ports} values, got {reference.size}"
            )

    scalar_z0 = _uniform_reference_or_none(reference)
    z0_out: float | np.ndarray = scalar_z0 if scalar_z0 is not None else reference.copy()

    s_params = np.zeros((n_ports, n_ports, n_freqs), dtype=np.complex128)
    indices = _pair_indices(
        n_ports,
        layout=resolved_layout,
        two_port_order=two_port_order,
    )
    for fi in range(n_freqs):
        for idx, (i, j) in enumerate(indices):
            s_params[i, j, fi] = s_data[fi][idx]

    return TouchstoneData(
        s_params=s_params,
        freqs=freqs,
        z0=z0_out,
        reference=reference,
        version=version,
        parameter=parameter,
        fmt=fmt,
        freq_unit=freq_unit,
        layout=resolved_layout,
        matrix_format=matrix_format,
        two_port_order=two_port_order,
        comments=tuple(comments),
        information=tuple(information),
    )


def network_quality_metrics(
    s_params: np.ndarray,
    *,
    reciprocity_tol: float = 1e-12,
    passivity_tol: float = 1e-12,
) -> dict[str, object]:
    """Return common RF network quality metrics for an S-matrix cube.

    The input must have shape ``(n_ports, n_ports, n_freqs)``.  Metrics are
    host-side diagnostics for reports and interop gates; they do not alter any
    solver path.
    """
    s = np.asarray(s_params)
    if s.ndim != 3 or s.shape[0] != s.shape[1]:
        raise ValueError("s_params must have shape (n_ports, n_ports, n_freqs)")

    n_ports, _, n_freqs = s.shape
    finite = bool(np.all(np.isfinite(s)))
    if not finite:
        return {
            "n_ports": int(n_ports),
            "n_freqs": int(n_freqs),
            "finite": False,
            "max_abs_s": float("nan"),
            "max_column_power": float("nan"),
            "max_singular_value": float("nan"),
            "passivity_excess": float("nan"),
            "reciprocity_error": float("nan"),
            "is_passive": False,
            "is_reciprocal": False,
        }

    max_column_power = 0.0
    max_singular_value = 0.0
    for fi in range(n_freqs):
        mat = s[:, :, fi]
        column_power = float(np.max(np.sum(np.abs(mat) ** 2, axis=0)))
        max_column_power = max(max_column_power, column_power)
        eig_max = float(np.linalg.eigvalsh(mat.conj().T @ mat).max())
        max_singular_value = max(max_singular_value, float(np.sqrt(max(eig_max, 0.0))))

    passivity_excess = max(0.0, max_singular_value**2 - 1.0)
    reciprocity_error = float(np.max(np.abs(s - np.swapaxes(s, 0, 1))))
    return {
        "n_ports": int(n_ports),
        "n_freqs": int(n_freqs),
        "finite": True,
        "max_abs_s": float(np.max(np.abs(s))) if s.size else 0.0,
        "max_column_power": float(max_column_power),
        "max_singular_value": float(max_singular_value),
        "passivity_excess": float(passivity_excess),
        "reciprocity_error": reciprocity_error,
        "is_passive": bool(passivity_excess <= passivity_tol),
        "is_reciprocal": bool(reciprocity_error <= reciprocity_tol),
    }


# =========================================================================
# Result export (HDF5)
# =========================================================================

def save_optimization_result(path, result, metadata=None, *,
                             include_latent=True,
                             include_density=True,
                             include_history=True):
    """Save OptimizeResult or TopologyResult to HDF5.

    Parameters
    ----------
    path : str or Path
    result : OptimizeResult or TopologyResult
    metadata : dict or None
        Extra metadata (grid shape, freq_max, etc.) stored as HDF5 attrs.
    include_latent : bool
        Save latent/density arrays (default True).
    include_density : bool
        Save projected density and PEC occupancy (default True).
    include_history : bool
        Save loss/beta history (default True).
    """
    import h5py

    path = Path(path)
    with h5py.File(path, "w") as f:
        f.create_dataset("eps_design", data=np.asarray(result.eps_design))

        if include_history:
            if hasattr(result, "loss_history"):
                f.create_dataset("loss_history", data=np.array(result.loss_history))
            elif hasattr(result, "history"):
                f.create_dataset("loss_history", data=np.array(result.history))
            if hasattr(result, "beta_history"):
                f.create_dataset("beta_history", data=np.array(result.beta_history))

        if include_latent and hasattr(result, "latent") and result.latent is not None:
            f.create_dataset("latent", data=np.asarray(result.latent))

        if include_density:
            if hasattr(result, "density") and result.density is not None:
                f.create_dataset("density", data=np.asarray(result.density))
            if hasattr(result, "density_projected") and result.density_projected is not None:
                f.create_dataset("density_projected", data=np.asarray(result.density_projected))
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
            # float(grid.dt): host-boundary — metadata written at save time, never inside a trace.
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

def save_simulation_dataset(path, sim, result, *,
                            include_fields=False,
                            include_time_series=True,
                            include_s_params=True,
                            include_config=True):
    """Save simulation input-output pair for surrogate model training.

    Creates an HDF5 file with selectable sections:
    - Input: simulation config and grid metadata
    - Output: time series, S-parameters
    - Fields: final 3D E/H snapshot (large, opt-in)

    Parameters
    ----------
    path : str or Path
    sim : Simulation
    result : Result
    include_fields : bool
        Save full 3D E/H field arrays (default False — large).
    include_time_series : bool
        Save probe time series (default True).
    include_s_params : bool
        Save S-parameters and frequencies (default True).
    include_config : bool
        Save simulation config as input metadata (default True).
    """
    import h5py

    path = Path(path)
    with h5py.File(path, "w") as f:
        # Input group: simulation config
        if include_config:
            inp = f.create_group("input")
            inp.attrs["freq_max"] = sim._freq_max
            inp.attrs["domain"] = sim._domain
            inp.attrs["boundary"] = sim._boundary
            if sim._dx is not None:
                inp.attrs["dx"] = sim._dx
            if hasattr(result, "grid") and result.grid is not None:
                grid = result.grid
                inp.attrs["grid_shape"] = grid.shape
                # float(grid.dt): host-boundary — HDF5 metadata, never inside a trace.
                inp.attrs["dt"] = float(grid.dt)

        # Output group
        out = f.create_group("output")

        if include_time_series:
            if hasattr(result, "time_series") and result.time_series is not None:
                ts = np.asarray(result.time_series)
                out.create_dataset("time_series", data=ts, compression="gzip")

        if include_s_params:
            if hasattr(result, "s_params") and result.s_params is not None:
                out.create_dataset("s_params", data=np.asarray(result.s_params))
            if hasattr(result, "freqs") and result.freqs is not None:
                out.create_dataset("freqs", data=np.asarray(result.freqs))

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
