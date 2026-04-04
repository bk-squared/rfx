"""Tests for Smith chart plotting.

Validates that plot_smith runs without error and returns Axes objects.
Uses matplotlib Agg backend to avoid display requirements.
"""

import numpy as np
import pytest

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


@pytest.mark.skipif(not HAS_MPL, reason="matplotlib not installed")
def test_smith_creates_figure():
    """plot_smith with minimal input should return an Axes."""
    from rfx.smith import plot_smith

    n = 50
    freqs = np.linspace(1e9, 10e9, n)
    # Random S11 inside the unit circle
    gamma = 0.5 * np.exp(1j * np.linspace(0, 2 * np.pi, n))

    ax = plot_smith(gamma, freqs)
    assert ax is not None
    assert hasattr(ax, "figure")
    assert hasattr(ax.figure, "savefig")
    plt.close(ax.figure)
    print("\ntest_smith_creates_figure: OK")


@pytest.mark.skipif(not HAS_MPL, reason="matplotlib not installed")
def test_smith_with_real_data():
    """Plot S11 = (Z - Z0) / (Z + Z0) for known impedances."""
    from rfx.smith import plot_smith

    z0 = 50.0
    # Sweep impedance from 10 to 200 ohm (real)
    z_real = np.linspace(10, 200, 100)
    s11 = (z_real - z0) / (z_real + z0)
    freqs = np.linspace(1e9, 5e9, 100)

    ax = plot_smith(s11.astype(complex), freqs, z0=z0)
    assert ax is not None

    # S11 for purely real impedance must be real-valued and lie on real axis
    assert np.allclose(s11.imag, 0.0)
    # At Z = Z0, S11 = 0 (chart centre)
    idx_match = np.argmin(np.abs(z_real - z0))
    assert abs(s11[idx_match]) < 0.05

    plt.close(ax.figure)

    # Now test complex impedance (R + jX)
    z_complex = 50 + 1j * np.linspace(-100, 100, 100)
    s11_complex = (z_complex - z0) / (z_complex + z0)
    freqs2 = np.linspace(1e9, 5e9, 100)
    ax2 = plot_smith(s11_complex, freqs2, z0=z0)
    assert ax2 is not None
    # All points should be inside or on the unit circle
    assert np.all(np.abs(s11_complex) <= 1.0 + 1e-10)
    plt.close(ax2.figure)
    print("\ntest_smith_with_real_data: OK")


@pytest.mark.skipif(not HAS_MPL, reason="matplotlib not installed")
def test_smith_markers():
    """Frequency markers should not raise and should add annotations."""
    from rfx.smith import plot_smith

    n = 80
    freqs = np.linspace(1e9, 10e9, n)
    gamma = 0.6 * np.exp(1j * np.linspace(0, np.pi, n))

    marker_freqs = [2e9, 5e9, 8e9]
    ax = plot_smith(gamma, freqs, markers=marker_freqs)
    assert ax is not None

    # Check that annotation texts were created
    texts = [t.get_text() for t in ax.texts]
    assert any("GHz" in t for t in texts), f"Expected GHz markers, got {texts}"
    plt.close(ax.figure)
    print("\ntest_smith_markers: OK")


@pytest.mark.skipif(not HAS_MPL, reason="matplotlib not installed")
def test_smith_custom_ax():
    """plot_smith should draw on a user-provided Axes."""
    from rfx.smith import plot_smith

    fig, ax_in = plt.subplots(figsize=(7, 7))
    n = 40
    freqs = np.linspace(2e9, 6e9, n)
    gamma = 0.3 * np.exp(1j * np.linspace(0, np.pi / 2, n))

    ax_out = plot_smith(gamma, freqs, ax=ax_in, show_vswr=False)
    # Must return the same Axes object
    assert ax_out is ax_in
    plt.close(fig)
    print("\ntest_smith_custom_ax: OK")
