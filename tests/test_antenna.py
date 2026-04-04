"""Tests for antenna metric extraction.

Validates gain, efficiency, beamwidth, front-to-back ratio, impedance
bandwidth, and the summary plot using synthetic far-field data.
"""

import numpy as np
import pytest

from rfx.farfield import FarFieldResult
from rfx.antenna import (
    antenna_gain,
    antenna_gain_dB,
    antenna_efficiency,
    half_power_beamwidth,
    front_to_back_ratio,
    antenna_bandwidth,
    BandwidthResult,
    plot_antenna_summary,
    _radiation_intensity,
    _total_radiated_power,
)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ---------------------------------------------------------------------------
# Helpers — synthetic far-field construction
# ---------------------------------------------------------------------------

def _make_isotropic_ff(n_theta=91, n_phi=181, n_freqs=1):
    """Create a FarFieldResult with uniform radiation (isotropic pattern).

    For an isotropic radiator, U = const everywhere, so G = 1 (0 dBi).
    We set E_theta = const, E_phi = 0 to keep things simple.

    The constant is chosen so that the integrated power gives the
    expected isotropic result: P_rad = 4*pi*U.
    """
    theta = np.linspace(0.01, np.pi - 0.01, n_theta)
    phi = np.linspace(0, 2 * np.pi - 2 * np.pi / n_phi, n_phi)
    freqs = np.array([3e9] * n_freqs, dtype=np.float64)

    # Constant |E| everywhere → isotropic
    E_theta = np.ones((n_freqs, n_theta, n_phi), dtype=np.complex128)
    E_phi = np.zeros((n_freqs, n_theta, n_phi), dtype=np.complex128)

    return FarFieldResult(
        E_theta=E_theta, E_phi=E_phi,
        theta=theta, phi=phi, freqs=freqs,
    )


def _make_dipole_ff(n_theta=181, n_phi=181, n_freqs=1):
    """Create a FarFieldResult with sin(theta) pattern (z-dipole).

    Short dipole: E_theta ~ sin(theta), E_phi = 0.
    Theoretical directivity = 1.5 (1.76 dBi).
    """
    theta = np.linspace(0.01, np.pi - 0.01, n_theta)
    phi = np.linspace(0, 2 * np.pi - 2 * np.pi / n_phi, n_phi)
    freqs = np.array([3e9] * n_freqs, dtype=np.float64)

    sin_th = np.sin(theta)
    E_theta = sin_th[None, :, None] * np.ones((n_freqs, n_theta, n_phi),
                                                dtype=np.complex128)
    E_phi = np.zeros((n_freqs, n_theta, n_phi), dtype=np.complex128)

    return FarFieldResult(
        E_theta=E_theta, E_phi=E_phi,
        theta=theta, phi=phi, freqs=freqs,
    )


def _make_directional_ff(n_theta=181, n_phi=181):
    """Create a FarFieldResult with a forward-directed beam (cos^4 pattern).

    Strong forward lobe (theta < 90 deg), weak backward radiation.
    """
    theta = np.linspace(0.01, np.pi - 0.01, n_theta)
    phi = np.linspace(0, 2 * np.pi - 2 * np.pi / n_phi, n_phi)
    freqs = np.array([3e9], dtype=np.float64)

    cos_th = np.cos(theta)
    # Forward lobe: cos^4(theta) for theta < pi/2, small constant behind
    pattern = np.where(cos_th > 0, cos_th ** 4, 0.01)
    E_theta = pattern[None, :, None] * np.ones((1, n_theta, n_phi),
                                                 dtype=np.complex128)
    E_phi = np.zeros((1, n_theta, n_phi), dtype=np.complex128)

    return FarFieldResult(
        E_theta=E_theta, E_phi=E_phi,
        theta=theta, phi=phi, freqs=freqs,
    )


# ---------------------------------------------------------------------------
# Tests: gain
# ---------------------------------------------------------------------------

def test_gain_isotropic():
    """Isotropic pattern should give gain = 1 (0 dBi) everywhere."""
    ff = _make_isotropic_ff()
    G = antenna_gain(ff)

    # For isotropic: G should be ~1.0 everywhere
    # Due to numerical integration on a finite grid, allow some tolerance
    G_mean = np.mean(G[0])
    assert abs(G_mean - 1.0) < 0.05, \
        f"Isotropic gain mean = {G_mean:.4f}, expected ~1.0"

    G_dB = antenna_gain_dB(ff)
    G_dB_mean = np.mean(G_dB[0])
    assert abs(G_dB_mean) < 0.3, \
        f"Isotropic gain = {G_dB_mean:.2f} dBi, expected ~0 dBi"
    print(f"\nIsotropic gain: {G_dB_mean:.3f} dBi (expected 0)")


def test_gain_dipole_peak():
    """Short dipole peak gain should be ~1.76 dBi."""
    ff = _make_dipole_ff()
    G = antenna_gain(ff)
    G_dB = 10.0 * np.log10(np.max(G[0]))

    # Theoretical: 1.76 dBi = 1.5 linear
    assert 1.0 < G_dB < 2.5, \
        f"Dipole peak gain = {G_dB:.2f} dBi, expected ~1.76 dBi"
    print(f"\nDipole peak gain: {G_dB:.3f} dBi (expected 1.76)")


def test_gain_realized():
    """Realized gain should be less than IEEE gain when input_power > P_rad."""
    ff = _make_dipole_ff()
    G_ieee = antenna_gain(ff)  # P_ref = P_rad
    P_rad = _total_radiated_power(ff)

    # Double the input power → realized gain should be half of IEEE gain
    G_real = antenna_gain(ff, input_power=P_rad * 2.0)
    np.testing.assert_allclose(G_real, G_ieee / 2.0, rtol=1e-10)
    print("\nRealized gain = IEEE gain / 2 when P_in = 2*P_rad: OK")


# ---------------------------------------------------------------------------
# Tests: efficiency
# ---------------------------------------------------------------------------

def test_efficiency_bounds():
    """Radiation efficiency should be in (0, 1] for P_in >= P_rad."""
    ff = _make_dipole_ff()
    P_rad = _total_radiated_power(ff)

    # P_in = P_rad → efficiency = 1.0
    eta_perfect = antenna_efficiency(ff, input_power=P_rad)
    np.testing.assert_allclose(eta_perfect[0], 1.0, rtol=1e-10)

    # P_in = 2*P_rad → efficiency = 0.5
    eta_half = antenna_efficiency(ff, input_power=P_rad * 2.0)
    np.testing.assert_allclose(eta_half[0], 0.5, rtol=1e-10)

    # General bounds
    eta_general = antenna_efficiency(ff, input_power=P_rad * 1.5)
    assert 0 < eta_general[0] < 1, \
        f"Efficiency = {eta_general[0]:.4f}, expected in (0, 1)"
    print(f"\nEfficiency bounds: eta={eta_general[0]:.4f} for P_in=1.5*P_rad")


# ---------------------------------------------------------------------------
# Tests: beamwidth
# ---------------------------------------------------------------------------

def test_beamwidth_dipole():
    """Dipole E-plane HPBW should be near 90 degrees (theoretical: 90 deg)."""
    ff = _make_dipole_ff(n_theta=361)
    hpbw = half_power_beamwidth(ff, plane="E")

    # Theoretical HPBW of short dipole in E-plane: 90 degrees
    assert 70 < hpbw < 110, \
        f"Dipole E-plane HPBW = {hpbw:.1f} deg, expected ~90 deg"
    print(f"\nDipole E-plane HPBW: {hpbw:.1f} deg (expected ~90)")


def test_beamwidth_symmetric():
    """Symmetric pattern should give equal E- and H-plane beamwidths.

    An isotropic-like pattern uniform across phi should have
    the same beamwidth in both planes.
    """
    # Dipole has same E_theta pattern for all phi → same HPBW in both cuts
    ff = _make_dipole_ff(n_theta=361, n_phi=181)
    hpbw_e = half_power_beamwidth(ff, plane="E")
    hpbw_h = half_power_beamwidth(ff, plane="H")

    # For our synthetic dipole with E_theta = sin(theta) independent of phi,
    # both planes should give the same HPBW
    assert abs(hpbw_e - hpbw_h) < 5.0, \
        f"E-plane HPBW={hpbw_e:.1f}, H-plane HPBW={hpbw_h:.1f}, diff > 5 deg"
    print(f"\nSymmetric HPBW: E={hpbw_e:.1f}, H={hpbw_h:.1f} deg")


def test_beamwidth_narrow():
    """Directional pattern should have narrow beamwidth."""
    ff = _make_directional_ff(n_theta=361)
    hpbw = half_power_beamwidth(ff, plane="E")

    # cos^4(theta) → HPBW should be relatively narrow
    assert 20 < hpbw < 80, \
        f"Directional HPBW = {hpbw:.1f} deg, expected 20-80 deg"
    print(f"\nDirectional E-plane HPBW: {hpbw:.1f} deg")


# ---------------------------------------------------------------------------
# Tests: front-to-back ratio
# ---------------------------------------------------------------------------

def test_front_to_back_directional():
    """Directional pattern should have positive F/B ratio."""
    ff = _make_directional_ff()
    fb = front_to_back_ratio(ff)

    # cos^4 in forward, 0.01 in backward → large F/B
    assert fb > 10, \
        f"F/B = {fb:.1f} dB, expected > 10 dB for directional pattern"
    print(f"\nDirectional F/B ratio: {fb:.1f} dB")


def test_front_to_back_symmetric():
    """Dipole (symmetric in theta) should have F/B near 0 dB."""
    ff = _make_dipole_ff()
    fb = front_to_back_ratio(ff)

    # sin(theta) pattern is symmetric about theta=pi/2 → F/B ~ 0
    assert abs(fb) < 3, \
        f"Dipole F/B = {fb:.1f} dB, expected ~0 dB"
    print(f"\nDipole F/B ratio: {fb:.1f} dB (expected ~0)")


# ---------------------------------------------------------------------------
# Tests: bandwidth
# ---------------------------------------------------------------------------

def test_bandwidth_extraction():
    """Synthetic S11 with a known -10 dB bandwidth centered at 3 GHz."""
    freqs = np.linspace(1e9, 5e9, 1001)
    # Create S11 with a dip at 3 GHz: Gaussian-shaped dip reaching -20 dB
    s11_db = -2.0 + (-18.0) * np.exp(-((freqs - 3e9) / (0.3e9)) ** 2)
    # Convert to complex S11 (add a phase so numpy recognizes it as complex)
    s11_mag = 10.0 ** (s11_db / 20.0)
    phase = np.linspace(0, 2 * np.pi, len(freqs))
    s11_complex = s11_mag * np.exp(1j * phase)

    bw = antenna_bandwidth(s11_complex, freqs, threshold_db=-10.0)
    assert isinstance(bw, BandwidthResult)

    # Verify centered near 3 GHz with reasonable bandwidth
    assert abs(bw.center_freq - 3e9) < 0.2e9, \
        f"Center freq = {bw.center_freq/1e9:.2f} GHz, expected ~3.0 GHz"
    assert bw.bandwidth_hz > 0.3e9, \
        f"Bandwidth = {bw.bandwidth_hz/1e6:.0f} MHz, expected > 300 MHz"
    assert bw.bandwidth_hz < 1.5e9, \
        f"Bandwidth = {bw.bandwidth_hz/1e6:.0f} MHz, expected < 1500 MHz"
    assert 0 < bw.fractional_bandwidth < 1, \
        f"FBW = {bw.fractional_bandwidth:.3f}, expected in (0, 1)"

    print(f"\nBandwidth: fc={bw.center_freq/1e9:.2f} GHz, "
          f"BW={bw.bandwidth_hz/1e6:.0f} MHz, FBW={bw.fractional_bandwidth:.1%}")


def test_bandwidth_from_db_array():
    """antenna_bandwidth should accept S11 already in dB (real array)."""
    freqs = np.linspace(1e9, 5e9, 501)
    s11_db = -2.0 + (-18.0) * np.exp(-((freqs - 3e9) / (0.3e9)) ** 2)

    bw = antenna_bandwidth(s11_db, freqs, threshold_db=-10.0)
    assert bw.bandwidth_hz > 0
    assert abs(bw.center_freq - 3e9) < 0.2e9
    print(f"\nBandwidth from dB array: BW={bw.bandwidth_hz/1e6:.0f} MHz")


def test_bandwidth_no_match():
    """When S11 is always above threshold, bandwidth should be zero."""
    freqs = np.linspace(1e9, 5e9, 101)
    s11_db = np.full_like(freqs, -3.0)  # always -3 dB, never below -10

    bw = antenna_bandwidth(s11_db, freqs, threshold_db=-10.0)
    assert bw.bandwidth_hz == 0.0
    assert np.isnan(bw.center_freq)
    print("\nNo-match bandwidth: OK")


# ---------------------------------------------------------------------------
# Tests: summary plot
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_MPL, reason="matplotlib not installed")
def test_summary_plot_basic():
    """plot_antenna_summary should return a Figure without errors."""
    ff = _make_dipole_ff(n_theta=37, n_phi=37)
    fig = plot_antenna_summary(ff)
    assert fig is not None
    assert hasattr(fig, "savefig")
    plt.close(fig)
    print("\nSummary plot (basic): OK")


@pytest.mark.skipif(not HAS_MPL, reason="matplotlib not installed")
def test_summary_plot_with_s11():
    """plot_antenna_summary with S11 data should create 3 panels."""
    ff = _make_dipole_ff(n_theta=37, n_phi=37)
    freqs = np.linspace(1e9, 5e9, 101)
    s11 = -5 - 15 * np.exp(-((freqs - 3e9) / 0.5e9) ** 2)

    fig = plot_antenna_summary(ff, s11=s11, freqs=freqs)
    assert fig is not None
    # Should have 3 axes (polar + rect gain + S11)
    axes = fig.get_axes()
    assert len(axes) == 3, f"Expected 3 panels, got {len(axes)}"
    plt.close(fig)
    print("\nSummary plot (with S11): OK, 3 panels")


# ---------------------------------------------------------------------------
# Tests: internal helpers
# ---------------------------------------------------------------------------

def test_radiation_intensity_positive():
    """Radiation intensity should be non-negative everywhere."""
    ff = _make_dipole_ff()
    U = _radiation_intensity(ff)
    assert np.all(U >= 0), "Radiation intensity must be non-negative"


def test_total_radiated_power_positive():
    """Total radiated power should be positive for a non-trivial pattern."""
    ff = _make_dipole_ff()
    P = _total_radiated_power(ff)
    assert P[0] > 0, f"P_rad = {P[0]}, expected > 0"
