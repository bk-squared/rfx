"""Tests for Touchstone (.sNp) file I/O.

Validates:
1. Write and read round-trip (RI format)
2. Write and read round-trip (MA format)
3. Write and read round-trip (DB format)
4. Port count inference from extension
5. Frequency unit handling
6. Multi-port support (.s2p, .s4p, .snp) with multi-line data blocks
7. DB format parsing for multi-port files
8. Frequency unit round-trip across all supported units
"""

import tempfile
import numpy as np
from pathlib import Path

from rfx.io import write_touchstone, read_touchstone


def _make_test_sparams(n_ports=2, n_freqs=10):
    """Generate random S-parameters for testing."""
    rng = np.random.default_rng(42)
    real = rng.standard_normal((n_ports, n_ports, n_freqs)) * 0.5
    imag = rng.standard_normal((n_ports, n_ports, n_freqs)) * 0.5
    s_params = real + 1j * imag
    freqs = np.linspace(1e9, 10e9, n_freqs)
    return s_params, freqs


def test_touchstone_roundtrip_ri():
    """Write and read back RI format — values should match."""
    s_params, freqs = _make_test_sparams()

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.s2p"
        write_touchstone(path, s_params, freqs, fmt="RI")
        s_read, f_read, z0 = read_touchstone(path)

    assert s_read.shape == s_params.shape
    assert f_read.shape == freqs.shape
    assert z0 == 50.0

    np.testing.assert_allclose(f_read, freqs, rtol=1e-6)
    np.testing.assert_allclose(s_read.real, s_params.real, atol=1e-6)
    np.testing.assert_allclose(s_read.imag, s_params.imag, atol=1e-6)

    print(f"\nRI round-trip: {s_params.shape} S-params, max err = "
          f"{np.max(np.abs(s_read - s_params)):.2e}")


def test_touchstone_roundtrip_ma():
    """Write and read back MA format."""
    s_params, freqs = _make_test_sparams()

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.s2p"
        write_touchstone(path, s_params, freqs, fmt="MA")
        s_read, f_read, z0 = read_touchstone(path)

    np.testing.assert_allclose(f_read, freqs, rtol=1e-6)
    np.testing.assert_allclose(np.abs(s_read), np.abs(s_params), rtol=1e-4)
    np.testing.assert_allclose(np.angle(s_read), np.angle(s_params), atol=1e-3)


def test_touchstone_roundtrip_db():
    """Write and read back DB format."""
    s_params, freqs = _make_test_sparams()

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.s2p"
        write_touchstone(path, s_params, freqs, fmt="DB")
        s_read, f_read, z0 = read_touchstone(path)

    np.testing.assert_allclose(f_read, freqs, rtol=1e-6)
    np.testing.assert_allclose(np.abs(s_read), np.abs(s_params), rtol=1e-3)


def test_touchstone_1port():
    """Single-port .s1p file."""
    s_params = np.array([[[0.5 + 0.1j, 0.3 - 0.2j, 0.1 + 0.05j]]])
    freqs = np.array([1e9, 5e9, 10e9])

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.s1p"
        write_touchstone(path, s_params, freqs, fmt="RI")
        s_read, f_read, z0 = read_touchstone(path)

    assert s_read.shape == (1, 1, 3)
    np.testing.assert_allclose(s_read, s_params, atol=1e-6)


def test_touchstone_freq_units():
    """Different frequency units should round-trip correctly."""
    s_params, freqs = _make_test_sparams(n_ports=1, n_freqs=5)

    for unit in ["Hz", "kHz", "MHz", "GHz"]:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.s1p"
            write_touchstone(path, s_params, freqs, freq_unit=unit, fmt="RI")
            s_read, f_read, z0 = read_touchstone(path)

        np.testing.assert_allclose(f_read, freqs, rtol=1e-5,
                                   err_msg=f"Freq unit {unit} failed")

    print("\nAll frequency units round-trip OK")


def test_touchstone_custom_z0():
    """Custom reference impedance should be preserved."""
    s_params, freqs = _make_test_sparams(n_ports=1, n_freqs=3)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.s1p"
        write_touchstone(path, s_params, freqs, z0=75.0, fmt="RI")
        _, _, z0 = read_touchstone(path)

    assert z0 == 75.0


def test_touchstone_reader_uses_touchstone_port_order():
    """Reader should parse standard S11, S21, S12, S22 ordering."""
    text = "\n".join([
        "! explicit 2-port ordering check",
        "# GHz S RI R 50",
        "1.0 11 0 21 0 12 0 22 0",
    ])

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "ordered.s2p"
        path.write_text(text)
        s_read, freqs, z0 = read_touchstone(path)

    assert z0 == 50.0
    np.testing.assert_allclose(freqs, np.array([1e9]))
    assert s_read[0, 0, 0] == 11 + 0j
    assert s_read[1, 0, 0] == 21 + 0j
    assert s_read[0, 1, 0] == 12 + 0j
    assert s_read[1, 1, 0] == 22 + 0j


def test_touchstone_writer_uses_touchstone_port_order():
    """Writer should emit standard S11, S21, S12, S22 ordering."""
    s_params = np.zeros((2, 2, 1), dtype=np.complex128)
    s_params[0, 0, 0] = 11 + 0j
    s_params[0, 1, 0] = 12 + 0j
    s_params[1, 0, 0] = 21 + 0j
    s_params[1, 1, 0] = 22 + 0j
    freqs = np.array([1e9])

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "ordered.s2p"
        write_touchstone(path, s_params, freqs, fmt="RI")
        data_line = [
            line for line in path.read_text().splitlines()
            if line and not line.startswith("!")
        ][1]

    tokens = data_line.split()
    assert tokens[0] == "1.000000000e+00"
    assert tokens[1:9] == [
        "1.100000000e+01", "0.000000000e+00",
        "2.100000000e+01", "0.000000000e+00",
        "1.200000000e+01", "0.000000000e+00",
        "2.200000000e+01", "0.000000000e+00",
    ]


# ---------------------------------------------------------------------------
# Multi-port tests (.s2p, .s3p, .s4p, .snp)
# ---------------------------------------------------------------------------

def test_write_read_s2p_roundtrip():
    """2-port RI round-trip: write then read should recover identical data."""
    s_params, freqs = _make_test_sparams(n_ports=2, n_freqs=20)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "filter.s2p"
        write_touchstone(path, s_params, freqs, fmt="RI")
        s_read, f_read, z0 = read_touchstone(path)

    assert s_read.shape == (2, 2, 20)
    assert z0 == 50.0
    np.testing.assert_allclose(f_read, freqs, rtol=1e-6)
    np.testing.assert_allclose(s_read, s_params, atol=1e-6)


def test_write_read_s3p_roundtrip():
    """3-port round-trip exercises multi-line data blocks (9 pairs > 4 per line)."""
    s_params, freqs = _make_test_sparams(n_ports=3, n_freqs=8)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "coupler.s3p"
        write_touchstone(path, s_params, freqs, fmt="RI")

        # Verify multi-line layout: each freq should span 3 lines
        # (first line: freq + 4 pairs, second: 4 pairs, third: 1 pair)
        content = path.read_text()
        s_read, f_read, z0 = read_touchstone(path)

    assert s_read.shape == (3, 3, 8)
    np.testing.assert_allclose(f_read, freqs, rtol=1e-6)
    np.testing.assert_allclose(s_read, s_params, atol=1e-6)

    # Check that continuation lines are present (indented, no freq column)
    data_lines = [
        l for l in content.splitlines()
        if l.strip() and not l.strip().startswith("!") and not l.strip().startswith("#")
    ]
    # 3-port: 9 pairs = first line (4 pairs) + 2 continuation lines (4+1)
    # So 3 raw lines per frequency, 8 freqs -> 24 data lines
    assert len(data_lines) == 8 * 3


def test_write_s4p():
    """4-port write produces valid multi-line Touchstone and round-trips."""
    s_params, freqs = _make_test_sparams(n_ports=4, n_freqs=5)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "diff_pair.s4p"
        write_touchstone(path, s_params, freqs, fmt="RI")

        content = path.read_text()
        s_read, f_read, z0 = read_touchstone(path)

    assert s_read.shape == (4, 4, 5)
    np.testing.assert_allclose(f_read, freqs, rtol=1e-6)
    np.testing.assert_allclose(s_read, s_params, atol=1e-6)

    # 4-port: 16 pairs. First line: freq + 4 pairs, then 3 cont lines of 4 pairs.
    # => 4 raw lines per frequency point.
    data_lines = [
        l for l in content.splitlines()
        if l.strip() and not l.strip().startswith("!") and not l.strip().startswith("#")
    ]
    assert len(data_lines) == 5 * 4


def test_write_read_s6p_roundtrip():
    """6-port (.s6p) — general N-port beyond standard .s4p."""
    s_params, freqs = _make_test_sparams(n_ports=6, n_freqs=3)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "network.s6p"
        write_touchstone(path, s_params, freqs, fmt="RI")
        s_read, f_read, z0 = read_touchstone(path)

    assert s_read.shape == (6, 6, 3)
    np.testing.assert_allclose(f_read, freqs, rtol=1e-6)
    np.testing.assert_allclose(s_read, s_params, atol=1e-6)


def test_read_s2p_db_format():
    """Read a hand-crafted 2-port DB-format file and verify decoded values."""
    # S11 = -20 dB @ -45 deg, S21 = -3 dB @ -90 deg
    # S12 = -3 dB @ -90 deg, S22 = -20 dB @ -45 deg
    text = "\n".join([
        "! 2-port filter in dB/angle format",
        "# GHz S DB R 50",
        "1.0  -20.0 -45.0  -3.0 -90.0  -3.0 -90.0  -20.0 -45.0",
        "2.0  -15.0 -30.0  -1.0 -60.0  -1.0 -60.0  -15.0 -30.0",
    ])

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "filter_db.s2p"
        path.write_text(text)
        s_read, freqs, z0 = read_touchstone(path)

    assert s_read.shape == (2, 2, 2)
    assert z0 == 50.0
    np.testing.assert_allclose(freqs, [1e9, 2e9])

    # S11 at f=1 GHz: mag = 10^(-20/20) = 0.1, angle = -45 deg
    s11_mag = np.abs(s_read[0, 0, 0])
    s11_ang = np.degrees(np.angle(s_read[0, 0, 0]))
    np.testing.assert_allclose(s11_mag, 0.1, rtol=1e-6)
    np.testing.assert_allclose(s11_ang, -45.0, atol=1e-3)

    # S21 at f=1 GHz: mag = 10^(-3/20) ~ 0.7079, angle = -90 deg
    s21_mag = np.abs(s_read[1, 0, 0])
    s21_ang = np.degrees(np.angle(s_read[1, 0, 0]))
    np.testing.assert_allclose(s21_mag, 10**(-3/20), rtol=1e-4)
    np.testing.assert_allclose(s21_ang, -90.0, atol=1e-3)


def test_frequency_units_multiport():
    """All four frequency units round-trip correctly for a 4-port file."""
    s_params, freqs = _make_test_sparams(n_ports=4, n_freqs=5)

    for unit in ["Hz", "kHz", "MHz", "GHz"]:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.s4p"
            write_touchstone(path, s_params, freqs, freq_unit=unit, fmt="RI")
            s_read, f_read, z0 = read_touchstone(path)

        np.testing.assert_allclose(f_read, freqs, rtol=1e-5,
                                   err_msg=f"Freq unit {unit} failed for 4-port")
        np.testing.assert_allclose(s_read, s_params, atol=1e-6,
                                   err_msg=f"S-param mismatch for unit {unit}")


def test_s4p_ma_roundtrip():
    """4-port MA (magnitude/angle) round-trip."""
    s_params, freqs = _make_test_sparams(n_ports=4, n_freqs=5)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.s4p"
        write_touchstone(path, s_params, freqs, fmt="MA")
        s_read, f_read, z0 = read_touchstone(path)

    np.testing.assert_allclose(np.abs(s_read), np.abs(s_params), rtol=1e-4)
    np.testing.assert_allclose(np.angle(s_read), np.angle(s_params), atol=1e-3)


def test_s4p_db_roundtrip():
    """4-port DB (dB/angle) round-trip."""
    s_params, freqs = _make_test_sparams(n_ports=4, n_freqs=5)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.s4p"
        write_touchstone(path, s_params, freqs, fmt="DB")
        s_read, f_read, z0 = read_touchstone(path)

    np.testing.assert_allclose(np.abs(s_read), np.abs(s_params), rtol=1e-3)


def test_read_s4p_multiline_handcrafted():
    """Read a hand-written 4-port file with explicit multi-line layout."""
    # 4-port: 16 pairs. Line 1: freq + 4 pairs, then 3 continuation lines.
    # Column-major order: S11 S21 S31 S41 | S12 S22 S32 S42 | S13 ... | S14 ...
    text = "\n".join([
        "! Hand-crafted 4-port RI",
        "# MHz S RI R 50",
        "100.0  1 0 2 0 3 0 4 0",
        "  5 0 6 0 7 0 8 0",
        "  9 0 10 0 11 0 12 0",
        "  13 0 14 0 15 0 16 0",
    ])

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "hand.s4p"
        path.write_text(text)
        s_read, freqs, z0 = read_touchstone(path)

    assert s_read.shape == (4, 4, 1)
    np.testing.assert_allclose(freqs, [100e6])

    # Verify column-major mapping:
    # vals 1..4 -> S11,S21,S31,S41 (column j=0)
    # vals 5..8 -> S12,S22,S32,S42 (column j=1)
    # etc.
    assert s_read[0, 0, 0] == 1 + 0j   # S11
    assert s_read[1, 0, 0] == 2 + 0j   # S21
    assert s_read[2, 0, 0] == 3 + 0j   # S31
    assert s_read[3, 0, 0] == 4 + 0j   # S41
    assert s_read[0, 1, 0] == 5 + 0j   # S12
    assert s_read[1, 1, 0] == 6 + 0j   # S22
    assert s_read[3, 3, 0] == 16 + 0j  # S44


def test_inline_comments_ignored():
    """Inline comments after data values should be stripped."""
    text = "\n".join([
        "# GHz S RI R 50",
        "1.0  0.5 0.1  0.8 -0.2  0.8 -0.2  0.5 0.1 ! some inline comment",
    ])

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "commented.s2p"
        path.write_text(text)
        s_read, freqs, z0 = read_touchstone(path)

    assert s_read.shape == (2, 2, 1)
    np.testing.assert_allclose(s_read[0, 0, 0], 0.5 + 0.1j, atol=1e-9)
