"""Tests for Touchstone (.sNp) file I/O.

Validates:
1. Write and read round-trip (RI format)
2. Write and read round-trip (MA format)
3. Write and read round-trip (DB format)
4. Port count inference from extension
5. Frequency unit handling
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

    print(f"\nAll frequency units round-trip OK")


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
