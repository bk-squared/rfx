"""Tests for S-parameter de-embedding utilities.

Validates:
1. Zero-length de-embedding leaves S-params unchanged
2. Half-wave port extension produces known analytical phase shift
3. Symmetric two-port de-embedding preserves reciprocity
4. Thru-line de-embedding recovers the DUT from fixture + DUT + fixture
"""

import numpy as np
import pytest

from rfx.deembed import deembed_port_extension, deembed_thru, _s_to_t, _t_to_s


# Speed of light — must match the module constant
_C0 = 299_792_458.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_lossless_line_s(theta: float | np.ndarray) -> np.ndarray:
    """S-matrix of an ideal lossless transmission line with electrical length theta.

    For a matched line (Z_line == Z0):
        S11 = S22 = 0
        S12 = S21 = exp(-j*theta)

    Parameters
    ----------
    theta : float or (n_freqs,) array
        Electrical length in radians.

    Returns
    -------
    (2, 2, n_freqs) complex array
    """
    theta = np.atleast_1d(theta)
    n = len(theta)
    s = np.zeros((2, 2, n), dtype=np.complex128)
    s[0, 1, :] = np.exp(-1j * theta)
    s[1, 0, :] = np.exp(-1j * theta)
    return s


def _cascade_two_port(s_a: np.ndarray, s_b: np.ndarray) -> np.ndarray:
    """Cascade two 2-port S-matrices via T-matrix multiplication."""
    n_freqs = s_a.shape[2]
    s_out = np.empty((2, 2, n_freqs), dtype=np.complex128)
    for fi in range(n_freqs):
        t_a = _s_to_t(s_a[:, :, fi])
        t_b = _s_to_t(s_b[:, :, fi])
        s_out[:, :, fi] = _t_to_s(t_a @ t_b)
    return s_out


# ---------------------------------------------------------------------------
# Tests: port extension de-embedding
# ---------------------------------------------------------------------------

class TestDeembedPortExtension:

    def test_zero_length_no_change(self):
        """De-embedding with zero port lengths must not alter S-params."""
        rng = np.random.default_rng(0)
        n_ports, n_freqs = 3, 20
        s = (rng.standard_normal((n_ports, n_ports, n_freqs))
             + 1j * rng.standard_normal((n_ports, n_ports, n_freqs)))
        freqs = np.linspace(1e9, 10e9, n_freqs)

        s_de = deembed_port_extension(s, freqs, [0.0, 0.0, 0.0])

        np.testing.assert_allclose(s_de, s, atol=1e-14)

    def test_half_wave_phase_shift(self):
        """De-embedding a half-wavelength line should flip S11 by 360 deg (no net change).

        For a single frequency f0 and a port extension of length L = lambda/2,
        the round-trip phase shift is 2*beta*L = 2*pi, so
        S11' = S11 * exp(j*2*pi) = S11.

        For a quarter-wave extension (round-trip = pi), S11' = -S11.
        """
        f0 = 3e9
        lam = _C0 / f0  # wavelength at f0
        freqs = np.array([f0])

        # Start with a known reflection
        s = np.array([[[0.5 + 0.3j]]])  # 1-port, 1 freq

        # --- Half-wave: round-trip = 2*pi => no change ---
        L_half = lam / 2.0
        s_de_half = deembed_port_extension(s, freqs, [L_half])
        np.testing.assert_allclose(s_de_half, s, atol=1e-12,
                                   err_msg="Half-wave de-embed should be identity")

        # --- Quarter-wave: round-trip = pi => S11' = S11 * exp(j*pi) = -S11 ---
        L_quarter = lam / 4.0
        s_de_quarter = deembed_port_extension(s, freqs, [L_quarter])
        np.testing.assert_allclose(s_de_quarter, -s, atol=1e-12,
                                   err_msg="Quarter-wave de-embed should negate S11")

    def test_half_wave_with_eps_eff(self):
        """Effective permittivity scales the electrical length correctly.

        If eps_eff = 4, the wave velocity is halved, so a physical length
        of lambda_0/4 becomes an electrical half-wave.
        """
        f0 = 3e9
        lam0 = _C0 / f0
        freqs = np.array([f0])
        s = np.array([[[0.7 - 0.2j]]])

        # Physical length = lam0/4, but eps_eff=4 => electrical length = lam0/2
        # Round-trip phase = 2*pi => identity
        L = lam0 / 4.0
        s_de = deembed_port_extension(s, freqs, [L], eps_eff=4.0)
        np.testing.assert_allclose(s_de, s, atol=1e-12)

    def test_symmetric_two_port(self):
        """De-embedding equal port lengths on a symmetric network preserves symmetry.

        A symmetric DUT (S11==S22, S12==S21) measured through identical feed
        lines should remain symmetric after de-embedding.
        """
        n_freqs = 30
        freqs = np.linspace(1e9, 6e9, n_freqs)

        # Construct a symmetric two-port with known phase
        rng = np.random.default_rng(42)
        s11 = 0.3 * np.exp(1j * rng.uniform(-np.pi, np.pi, n_freqs))
        s21 = 0.8 * np.exp(1j * rng.uniform(-np.pi, np.pi, n_freqs))

        s = np.zeros((2, 2, n_freqs), dtype=np.complex128)
        s[0, 0, :] = s11
        s[1, 1, :] = s11  # symmetric
        s[0, 1, :] = s21
        s[1, 0, :] = s21  # reciprocal

        L = 0.01  # 1 cm feed lines on both ports
        s_de = deembed_port_extension(s, freqs, [L, L])

        # Check symmetry is preserved
        np.testing.assert_allclose(s_de[0, 0], s_de[1, 1], atol=1e-14,
                                   err_msg="S11 != S22 after symmetric de-embed")
        np.testing.assert_allclose(s_de[0, 1], s_de[1, 0], atol=1e-14,
                                   err_msg="S12 != S21 after symmetric de-embed")

        # The magnitudes should be unchanged (phase-only correction)
        np.testing.assert_allclose(np.abs(s_de), np.abs(s), atol=1e-14,
                                   err_msg="Magnitude changed by phase-only de-embed")

    def test_port_length_mismatch_raises(self):
        """Mismatched port count should raise ValueError."""
        s = np.zeros((2, 2, 5), dtype=np.complex128)
        freqs = np.linspace(1e9, 5e9, 5)

        with pytest.raises(ValueError, match="port_lengths has 3 entries"):
            deembed_port_extension(s, freqs, [0.01, 0.01, 0.01])

    def test_roundtrip_embed_deembed(self):
        """Embedding then de-embedding should recover the original S-params.

        Simulate measuring a DUT through matched feed lines:
        S_meas_ij = S_dut_ij * exp(-j*beta*(L_i + L_j))
        Then de-embedding should recover S_dut.
        """
        n_freqs = 50
        freqs = np.linspace(1e9, 10e9, n_freqs)
        beta = 2.0 * np.pi * freqs / _C0

        # Original DUT S-params
        rng = np.random.default_rng(99)
        s_dut = 0.5 * (rng.standard_normal((2, 2, n_freqs))
                        + 1j * rng.standard_normal((2, 2, n_freqs)))

        # Simulate the effect of feed lines (embedding)
        L = np.array([0.015, 0.020])  # different port lengths
        s_meas = np.empty_like(s_dut)
        for i in range(2):
            for j in range(2):
                s_meas[i, j, :] = s_dut[i, j, :] * np.exp(-1j * beta * (L[i] + L[j]))

        # De-embed
        s_recovered = deembed_port_extension(s_meas, freqs, L)

        np.testing.assert_allclose(s_recovered, s_dut, atol=1e-12,
                                   err_msg="Round-trip embed/de-embed failed")


# ---------------------------------------------------------------------------
# Tests: thru-line de-embedding
# ---------------------------------------------------------------------------

class TestDeembedThru:

    def test_thru_deembed_identity(self):
        """If the DUT is a thru and the fixture is a thru, result should be thru."""
        n_freqs = 20
        np.linspace(1e9, 10e9, n_freqs)

        # Ideal thru: S11=S22=0, S12=S21=1
        s_thru = np.zeros((2, 2, n_freqs), dtype=np.complex128)
        s_thru[0, 1, :] = 1.0
        s_thru[1, 0, :] = 1.0

        # Measured = thru (fixture is ideal => no effect)
        s_meas = s_thru.copy()

        s_dut = deembed_thru(s_meas, s_thru)

        np.testing.assert_allclose(np.abs(s_dut[0, 1]), 1.0, atol=1e-10,
                                   err_msg="Thru de-embed of thru should give |S21|=1")
        np.testing.assert_allclose(np.abs(s_dut[0, 0]), 0.0, atol=1e-10,
                                   err_msg="Thru de-embed of thru should give |S11|=0")

    def test_thru_deembed_recovers_dut(self):
        """Cascade fixture + DUT + fixture, then de-embed with fixture thru.

        The thru standard is fixture cascaded with fixture.
        De-embedding the measurement should recover the DUT.
        """
        n_freqs = 30
        freqs = np.linspace(1e9, 6e9, n_freqs)
        theta_fix = 2.0 * np.pi * freqs * 0.01 / _C0  # 1 cm fixture

        # Fixture: a lossless transmission line
        s_fixture = _make_lossless_line_s(theta_fix)

        # DUT: a known attenuator (symmetric, reciprocal)
        atten = 0.7  # |S21| = 0.7
        s_dut_true = np.zeros((2, 2, n_freqs), dtype=np.complex128)
        s_dut_true[0, 1, :] = atten
        s_dut_true[1, 0, :] = atten
        # Small reflection
        s_dut_true[0, 0, :] = 0.1
        s_dut_true[1, 1, :] = 0.1

        # Measured = fixture + DUT + fixture
        s_meas = _cascade_two_port(
            _cascade_two_port(s_fixture, s_dut_true),
            s_fixture,
        )

        # Thru standard = fixture + fixture
        s_thru = _cascade_two_port(s_fixture, s_fixture)

        # De-embed
        s_dut_recovered = deembed_thru(s_meas, s_thru)

        np.testing.assert_allclose(s_dut_recovered, s_dut_true, atol=1e-10,
                                   err_msg="Thru de-embed did not recover DUT")

    def test_thru_deembed_input_validation(self):
        """Non-2-port inputs should raise ValueError."""
        s3 = np.zeros((3, 3, 5), dtype=np.complex128)
        s2 = np.zeros((2, 2, 5), dtype=np.complex128)

        with pytest.raises(ValueError, match="s_measured must be a 2-port"):
            deembed_thru(s3, s2)

        with pytest.raises(ValueError, match="s_thru must be a 2-port"):
            deembed_thru(s2, s3)

    def test_thru_deembed_freq_mismatch(self):
        """Mismatched frequency counts should raise ValueError."""
        s_a = np.zeros((2, 2, 10), dtype=np.complex128)
        s_b = np.zeros((2, 2, 5), dtype=np.complex128)

        with pytest.raises(ValueError, match="Frequency count mismatch"):
            deembed_thru(s_a, s_b)


# ---------------------------------------------------------------------------
# Tests: S <-> T matrix round-trip
# ---------------------------------------------------------------------------

class TestSTConversion:

    def test_s_to_t_roundtrip(self):
        """S -> T -> S should be identity."""
        rng = np.random.default_rng(7)
        for _ in range(10):
            s = rng.standard_normal((2, 2)) + 1j * rng.standard_normal((2, 2))
            # Ensure S21 != 0 for invertibility
            s[1, 0] = max(abs(s[1, 0]), 0.1) * np.exp(1j * np.angle(s[1, 0]))
            t = _s_to_t(s)
            s_back = _t_to_s(t)
            np.testing.assert_allclose(s_back, s, atol=1e-12)
