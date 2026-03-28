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
