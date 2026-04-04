"""S-parameter de-embedding utilities.

Provides post-processing routines to shift reference planes and remove
unwanted fixture or feed-line effects from measured or simulated S-parameters.

Two methods are implemented:

1. **Port extension de-embedding** — removes the phase delay introduced by
   known lengths of transmission line at each port.  This is the simplest
   and most common de-embedding approach for simulation post-processing.

2. **Thru-line de-embedding** — uses a measured thru standard to remove
   symmetric fixture effects from a DUT measurement (TRL-lite / thru-only
   calibration).
"""

from __future__ import annotations

import numpy as np

# Speed of light in vacuum (m/s)
_C0 = 299_792_458.0


def deembed_port_extension(
    s_matrix: np.ndarray,
    freqs: np.ndarray,
    port_lengths: list[float] | np.ndarray,
    z0: float = 50.0,
    eps_eff: float | np.ndarray = 1.0,
) -> np.ndarray:
    """Remove port extension (transmission line) effects from S-parameters.

    Shifts the reference plane inward by removing the phase delay of the
    feed lines at each port.  Assumes lossless, matched feed lines (same
    impedance as the port reference impedance) so only a phase correction
    is needed.

    Parameters
    ----------
    s_matrix : ndarray, shape (n_ports, n_ports, n_freqs)
        Complex S-parameter matrix.
    freqs : ndarray, shape (n_freqs,)
        Frequency points in Hz.
    port_lengths : list of float or ndarray, shape (n_ports,)
        Physical length to de-embed from each port, in metres.
    z0 : float
        Port reference impedance (ohm).  Not used in the phase-only
        correction but kept for API consistency and future lossy extension.
    eps_eff : float or ndarray
        Effective relative permittivity of the feed line.  A scalar
        applies uniformly; an array of shape ``(n_freqs,)`` supports
        dispersive media.

    Returns
    -------
    ndarray, shape (n_ports, n_ports, n_freqs)
        De-embedded S-parameter matrix.
    """
    s_matrix = np.asarray(s_matrix, dtype=np.complex128)
    freqs = np.asarray(freqs, dtype=np.float64)
    port_lengths = np.asarray(port_lengths, dtype=np.float64)
    eps_eff = np.asarray(eps_eff, dtype=np.float64)

    n_ports = s_matrix.shape[0]
    s_matrix.shape[2]

    if port_lengths.shape[0] != n_ports:
        raise ValueError(
            f"port_lengths has {port_lengths.shape[0]} entries but "
            f"s_matrix has {n_ports} ports"
        )

    # Propagation constant: beta = 2*pi*f * sqrt(eps_eff) / c
    # Shape: (n_freqs,) after broadcast
    beta = 2.0 * np.pi * freqs * np.sqrt(eps_eff) / _C0  # (n_freqs,)

    # Phase shift per port: exp(+j * beta * L_i)
    # We *add* phase to undo the delay (shift reference plane inward).
    # port_phase[i, f] = exp(j * beta[f] * L_i)
    port_phase = np.exp(1j * np.outer(port_lengths, beta))  # (n_ports, n_freqs)

    # Build the de-embedding matrix.
    # S'_ij(f) = S_ij(f) * exp(j * beta * L_i) * exp(j * beta * L_j)
    # For diagonal (reflection): round-trip  => exp(j * 2*beta*L_i)
    # For off-diagonal (transmission): one-way each => exp(j*beta*(L_i + L_j))
    # Both are captured by the product port_phase[i] * port_phase[j].
    s_deembedded = np.empty_like(s_matrix)
    for i in range(n_ports):
        for j in range(n_ports):
            s_deembedded[i, j, :] = (
                s_matrix[i, j, :] * port_phase[i, :] * port_phase[j, :]
            )

    return s_deembedded


def deembed_thru(
    s_measured: np.ndarray,
    s_thru: np.ndarray,
) -> np.ndarray:
    """De-embed a DUT using a thru-line measurement (TRL-lite).

    Assumes a symmetric two-port fixture: the thru standard captures the
    combined effect of both fixture halves in cascade.  The fixture
    half-network is extracted from the thru, inverted, and removed from
    each side of the measured DUT.

    This is a simplified single-thru calibration; for full TRL accuracy,
    reflect and line standards would also be needed.

    The algorithm:

    1. Convert the thru S-matrix to a transfer (T) matrix.
    2. Take the matrix square root to get the fixture half T-matrix.
    3. Invert the fixture half.
    4. Cascade: T_dut = T_fix_inv @ T_meas @ T_fix_inv.
    5. Convert back to S-parameters.

    Parameters
    ----------
    s_measured : ndarray, shape (2, 2, n_freqs)
        Measured S-parameters of the DUT including fixture.
    s_thru : ndarray, shape (2, 2, n_freqs)
        Measured S-parameters of the thru standard (fixture back-to-back).

    Returns
    -------
    ndarray, shape (2, 2, n_freqs)
        De-embedded S-parameters of the DUT.

    Raises
    ------
    ValueError
        If inputs are not 2-port matrices or have mismatched frequency counts.
    """
    s_measured = np.asarray(s_measured, dtype=np.complex128)
    s_thru = np.asarray(s_thru, dtype=np.complex128)

    if s_measured.shape[:2] != (2, 2):
        raise ValueError("s_measured must be a 2-port S-matrix (2, 2, n_freqs)")
    if s_thru.shape[:2] != (2, 2):
        raise ValueError("s_thru must be a 2-port S-matrix (2, 2, n_freqs)")
    if s_measured.shape[2] != s_thru.shape[2]:
        raise ValueError(
            f"Frequency count mismatch: s_measured has {s_measured.shape[2]}, "
            f"s_thru has {s_thru.shape[2]}"
        )

    n_freqs = s_measured.shape[2]
    s_dut = np.empty_like(s_measured)

    for fi in range(n_freqs):
        t_meas = _s_to_t(s_measured[:, :, fi])
        t_thru = _s_to_t(s_thru[:, :, fi])

        # Matrix square root of the thru T-matrix gives the fixture half
        t_fix_half = _matrix_sqrt_2x2(t_thru)

        # Invert the fixture half
        t_fix_inv = np.linalg.inv(t_fix_half)

        # De-embed: remove fixture from both sides
        t_dut = t_fix_inv @ t_meas @ t_fix_inv

        s_dut[:, :, fi] = _t_to_s(t_dut)

    return s_dut


# ---------------------------------------------------------------------------
# Internal helpers: S <-> T matrix conversions for 2-port networks
# ---------------------------------------------------------------------------

def _s_to_t(s: np.ndarray) -> np.ndarray:
    """Convert a 2x2 S-matrix to a transfer (T) matrix.

    T-matrix definition (wave-transfer / chain-scattering):
        [b1]   [T11  T12] [a2]
        [a1] = [T21  T22] [b2]

    Conversion:
        T11 = -(S11*S22 - S12*S21) / S21
        T12 =  S11 / S21
        T21 = -S22 / S21
        T22 =  1 / S21
    """
    s11, s12, s21, s22 = s[0, 0], s[0, 1], s[1, 0], s[1, 1]
    det_s = s11 * s22 - s12 * s21
    t = np.array([
        [-det_s / s21, s11 / s21],
        [-s22 / s21, 1.0 / s21],
    ], dtype=np.complex128)
    return t


def _t_to_s(t: np.ndarray) -> np.ndarray:
    """Convert a 2x2 transfer (T) matrix back to an S-matrix.

    Inverse of ``_s_to_t``:
        S11 =  T12 / T22
        S12 =  (T11*T22 - T12*T21) / T22
        S21 =  1 / T22
        S22 = -T21 / T22
    """
    t11, t12, t21, t22 = t[0, 0], t[0, 1], t[1, 0], t[1, 1]
    det_t = t11 * t22 - t12 * t21
    s = np.array([
        [t12 / t22, det_t / t22],
        [1.0 / t22, -t21 / t22],
    ], dtype=np.complex128)
    return s


def _matrix_sqrt_2x2(m: np.ndarray) -> np.ndarray:
    """Compute a matrix square root of a 2x2 matrix.

    Uses the Cayley-Hamilton closed-form: for a 2x2 matrix M,
        sqrt(M) = (M + sqrt(det(M)) * I) / s
    where s = sqrt(trace(M) + 2*sqrt(det(M))).

    Chooses the principal branch (positive real part of s).
    """
    det_m = np.linalg.det(m)
    sqrt_det = np.sqrt(det_m)
    tr = np.trace(m)

    s = np.sqrt(tr + 2.0 * sqrt_det)

    # Guard against near-zero s (degenerate case)
    if abs(s) < 1e-30:
        # Fall back to eigendecomposition
        eigvals, eigvecs = np.linalg.eig(m)
        sqrt_m = eigvecs @ np.diag(np.sqrt(eigvals)) @ np.linalg.inv(eigvecs)
        return sqrt_m

    sqrt_m = (m + sqrt_det * np.eye(2, dtype=np.complex128)) / s
    return sqrt_m
