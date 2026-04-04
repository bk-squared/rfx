"""Antenna metric extraction from far-field data.

Provides gain, efficiency, beamwidth, front-to-back ratio, impedance
bandwidth, and a multi-panel summary plot.  All functions accept the
:class:`~rfx.farfield.FarFieldResult` produced by
:func:`~rfx.farfield.compute_far_field`.

References:
    Balanis, "Antenna Theory: Analysis and Design", 4th ed.
    IEEE Std 145-2013, "Definitions of Terms for Antennas"
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from rfx.farfield import FarFieldResult
from rfx.core.yee import EPS_0, MU_0

ETA_0 = float(np.sqrt(MU_0 / EPS_0))  # ~377 ohm


# ---------------------------------------------------------------------------
# Radiation intensity & radiated power helpers
# ---------------------------------------------------------------------------

def _radiation_intensity(ff: FarFieldResult) -> np.ndarray:
    """Radiation intensity U(theta, phi) for each frequency.

    U = (|E_theta|^2 + |E_phi|^2) / (2 * eta_0)

    The far-field result stores E*r products (V*m), so
    U = r^2 * |E|^2 / (2*eta_0) but we work per unit r^2
    since the 1/r factor is already omitted.

    Returns (n_freqs, n_theta, n_phi).
    """
    return (np.abs(ff.E_theta) ** 2 + np.abs(ff.E_phi) ** 2) / (2.0 * ETA_0)


def _total_radiated_power(ff: FarFieldResult) -> np.ndarray:
    """Total radiated power by integrating U over the full sphere.

    P_rad = integral U(theta, phi) sin(theta) d_theta d_phi

    Returns (n_freqs,).
    """
    U = _radiation_intensity(ff)
    theta = ff.theta
    dth = np.gradient(theta) if len(theta) > 1 else np.array([np.pi])
    dph = np.gradient(ff.phi) if len(ff.phi) > 1 else np.array([2 * np.pi])
    sin_th = np.sin(theta)

    integrand = U * sin_th[None, :, None]
    P_rad = np.sum(integrand * dth[None, :, None] * dph[None, None, :],
                   axis=(1, 2))
    return P_rad


# ---------------------------------------------------------------------------
# Antenna gain
# ---------------------------------------------------------------------------

def antenna_gain(
    ff: FarFieldResult,
    *,
    input_power: np.ndarray | float | None = None,
) -> np.ndarray:
    """Antenna gain G(theta, phi) for each frequency.

    G = 4*pi * U(theta, phi) / P_ref

    where P_ref = P_rad (IEEE gain) when *input_power* is ``None``,
    or P_ref = P_in (realized gain) when *input_power* is given.

    Parameters
    ----------
    ff : FarFieldResult
    input_power : array-like (n_freqs,) or scalar, optional
        Total input power in watts.  When provided the result is
        *realized gain* which accounts for mismatch loss.

    Returns
    -------
    gain : (n_freqs, n_theta, n_phi) array (linear, not dB)
    """
    U = _radiation_intensity(ff)
    if input_power is not None:
        P_ref = np.broadcast_to(
            np.asarray(input_power, dtype=np.float64),
            (len(ff.freqs),),
        )
    else:
        P_ref = _total_radiated_power(ff)

    safe_P = np.where(P_ref > 0, P_ref, 1e-30)
    return 4.0 * np.pi * U / safe_P[:, None, None]


def antenna_gain_dB(
    ff: FarFieldResult,
    *,
    input_power: np.ndarray | float | None = None,
) -> np.ndarray:
    """Antenna gain in dBi.  Same interface as :func:`antenna_gain`."""
    G = antenna_gain(ff, input_power=input_power)
    return 10.0 * np.log10(np.maximum(G, 1e-30))


# ---------------------------------------------------------------------------
# Radiation efficiency
# ---------------------------------------------------------------------------

def antenna_efficiency(
    ff: FarFieldResult,
    input_power: np.ndarray | float,
) -> np.ndarray:
    """Radiation efficiency eta = P_rad / P_in.

    Parameters
    ----------
    ff : FarFieldResult
    input_power : array-like (n_freqs,) or scalar
        Total input power in watts.

    Returns
    -------
    eta : (n_freqs,) array, values in [0, inf) theoretically in (0, 1].
    """
    P_rad = _total_radiated_power(ff)
    P_in = np.broadcast_to(
        np.asarray(input_power, dtype=np.float64),
        (len(ff.freqs),),
    )
    safe_P_in = np.where(P_in > 0, P_in, 1e-30)
    return P_rad / safe_P_in


# ---------------------------------------------------------------------------
# Half-power beamwidth (HPBW)
# ---------------------------------------------------------------------------

def half_power_beamwidth(
    ff: FarFieldResult,
    *,
    plane: str = "E",
    freq_idx: int = 0,
) -> float:
    """Half-power (3 dB) beamwidth in degrees.

    Parameters
    ----------
    ff : FarFieldResult
    plane : ``"E"`` (phi=0 cut) or ``"H"`` (phi=pi/2 cut).
    freq_idx : frequency index.

    Returns
    -------
    hpbw : float, degrees.  Returns ``nan`` if the 3 dB points
        cannot be found (e.g. omnidirectional in the cut plane).
    """
    if plane.upper() == "E":
        phi_idx = 0
    elif plane.upper() == "H":
        # Find the phi index closest to pi/2
        phi_idx = int(np.argmin(np.abs(ff.phi - np.pi / 2)))
    else:
        raise ValueError(f"plane must be 'E' or 'H', got {plane!r}")

    power = (np.abs(ff.E_theta[freq_idx, :, phi_idx]) ** 2
             + np.abs(ff.E_phi[freq_idx, :, phi_idx]) ** 2)
    peak = np.max(power)
    if peak <= 0:
        return float("nan")

    power_norm = power / peak
    half_power = 0.5  # -3 dB in linear

    theta_deg = np.degrees(ff.theta)

    # Find where the pattern crosses the half-power level
    above = power_norm >= half_power
    if not np.any(above):
        return float("nan")

    # Find first and last indices above the half-power level
    indices = np.where(above)[0]
    if len(indices) < 2:
        return float("nan")

    # Interpolate the 3-dB crossing points
    left_idx = indices[0]
    right_idx = indices[-1]

    # Left edge interpolation
    if left_idx > 0:
        x0, x1 = theta_deg[left_idx - 1], theta_deg[left_idx]
        y0, y1 = power_norm[left_idx - 1], power_norm[left_idx]
        if y1 != y0:
            theta_left = x0 + (half_power - y0) * (x1 - x0) / (y1 - y0)
        else:
            theta_left = x0
    else:
        theta_left = theta_deg[left_idx]

    # Right edge interpolation
    if right_idx < len(theta_deg) - 1:
        x0, x1 = theta_deg[right_idx], theta_deg[right_idx + 1]
        y0, y1 = power_norm[right_idx], power_norm[right_idx + 1]
        if y1 != y0:
            theta_right = x0 + (half_power - y0) * (x1 - x0) / (y1 - y0)
        else:
            theta_right = x1
    else:
        theta_right = theta_deg[right_idx]

    return float(theta_right - theta_left)


# ---------------------------------------------------------------------------
# Front-to-back ratio
# ---------------------------------------------------------------------------

def front_to_back_ratio(
    ff: FarFieldResult,
    *,
    freq_idx: int = 0,
) -> float:
    """Front-to-back ratio in dB.

    F/B = max gain in forward hemisphere (theta < 90 deg)
        - max gain in backward hemisphere (theta > 90 deg)

    Parameters
    ----------
    ff : FarFieldResult
    freq_idx : frequency index.

    Returns
    -------
    fb_dB : float
    """
    power = (np.abs(ff.E_theta[freq_idx]) ** 2
             + np.abs(ff.E_phi[freq_idx]) ** 2)  # (n_theta, n_phi)
    theta = ff.theta

    fwd_mask = theta < np.pi / 2
    bwd_mask = theta > np.pi / 2

    if not np.any(fwd_mask) or not np.any(bwd_mask):
        return float("nan")

    fwd_max = np.max(power[fwd_mask, :])
    bwd_max = np.max(power[bwd_mask, :])

    if bwd_max <= 0:
        return float("inf")
    if fwd_max <= 0:
        return float("-inf")

    return float(10.0 * np.log10(fwd_max / bwd_max))


# ---------------------------------------------------------------------------
# Impedance bandwidth from S11
# ---------------------------------------------------------------------------

class BandwidthResult(NamedTuple):
    """Impedance bandwidth extraction result.

    center_freq : float, Hz
    bandwidth_hz : float, Hz
    fractional_bandwidth : float (ratio, e.g. 0.10 = 10 %)
    freq_lo : float, Hz — lower edge
    freq_hi : float, Hz — upper edge
    """
    center_freq: float
    bandwidth_hz: float
    fractional_bandwidth: float
    freq_lo: float
    freq_hi: float


def antenna_bandwidth(
    s11: np.ndarray,
    freqs: np.ndarray,
    *,
    threshold_db: float = -10.0,
) -> BandwidthResult:
    """Impedance bandwidth where |S11| < *threshold_db*.

    Finds the widest contiguous frequency range below the threshold.

    Parameters
    ----------
    s11 : (n_freqs,) complex or real.
        If complex, magnitude is computed.  If real, assumed to be in dB.
    freqs : (n_freqs,) Hz.
    threshold_db : float
        Threshold in dB (default -10 dB).

    Returns
    -------
    BandwidthResult
    """
    freqs = np.asarray(freqs, dtype=np.float64)
    s11 = np.asarray(s11)

    if np.iscomplexobj(s11):
        s11_db = 20.0 * np.log10(np.maximum(np.abs(s11), 1e-30))
    else:
        s11_db = s11.astype(np.float64)

    below = s11_db < threshold_db

    if not np.any(below):
        return BandwidthResult(
            center_freq=float("nan"),
            bandwidth_hz=0.0,
            fractional_bandwidth=0.0,
            freq_lo=float("nan"),
            freq_hi=float("nan"),
        )

    # Find the widest contiguous run below threshold
    best_start, best_len = 0, 0
    cur_start, cur_len = 0, 0
    for i, b in enumerate(below):
        if b:
            if cur_len == 0:
                cur_start = i
            cur_len += 1
            if cur_len > best_len:
                best_start = cur_start
                best_len = cur_len
        else:
            cur_len = 0

    # Interpolate edges for better accuracy
    lo_idx = best_start
    hi_idx = best_start + best_len - 1

    # Left edge
    if lo_idx > 0:
        f0, f1 = freqs[lo_idx - 1], freqs[lo_idx]
        d0, d1 = s11_db[lo_idx - 1], s11_db[lo_idx]
        if d1 != d0:
            freq_lo = f0 + (threshold_db - d0) * (f1 - f0) / (d1 - d0)
        else:
            freq_lo = f0
    else:
        freq_lo = freqs[lo_idx]

    # Right edge
    if hi_idx < len(freqs) - 1:
        f0, f1 = freqs[hi_idx], freqs[hi_idx + 1]
        d0, d1 = s11_db[hi_idx], s11_db[hi_idx + 1]
        if d1 != d0:
            freq_hi = f0 + (threshold_db - d0) * (f1 - f0) / (d1 - d0)
        else:
            freq_hi = f1
    else:
        freq_hi = freqs[hi_idx]

    bw = freq_hi - freq_lo
    fc = (freq_lo + freq_hi) / 2.0
    fbw = bw / fc if fc > 0 else 0.0

    return BandwidthResult(
        center_freq=float(fc),
        bandwidth_hz=float(bw),
        fractional_bandwidth=float(fbw),
        freq_lo=float(freq_lo),
        freq_hi=float(freq_hi),
    )


# ---------------------------------------------------------------------------
# Summary plot
# ---------------------------------------------------------------------------

def plot_antenna_summary(
    ff: FarFieldResult,
    *,
    s11: np.ndarray | None = None,
    freqs: np.ndarray | None = None,
    freq_idx: int = 0,
    input_power: np.ndarray | float | None = None,
) -> object:
    """Multi-panel antenna summary figure.

    Panels:
    1. Radiation pattern (polar, E-plane and H-plane)
    2. Gain vs theta (rectangular)
    3. S11 vs frequency (if provided)
    4. Efficiency vs frequency or gain statistics

    Parameters
    ----------
    ff : FarFieldResult
    s11 : (n_freqs,) complex or real, optional
    freqs : (n_freqs,) Hz for the S11 data, optional
    freq_idx : frequency index for pattern plots.
    input_power : for realized gain computation, optional.

    Returns
    -------
    matplotlib Figure
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib is required for plot_antenna_summary")

    has_s11 = s11 is not None and freqs is not None
    n_panels = 3 if has_s11 else 2

    fig = plt.figure(figsize=(5 * n_panels, 5))

    # --- Panel 1: Polar radiation pattern ---
    ax1 = fig.add_subplot(1, n_panels, 1, projection="polar")

    G = antenna_gain(ff, input_power=input_power)  # (nf, nth, nph)
    G_freq = G[freq_idx]  # (nth, nph)

    # E-plane (phi=0)
    G_e = G_freq[:, 0]
    G_e_dB = 10.0 * np.log10(np.maximum(G_e, 1e-30))
    G_e_plot = np.maximum(G_e_dB, np.max(G_e_dB) - 40) - (np.max(G_e_dB) - 40)

    ax1.plot(ff.theta, G_e_plot, label="E-plane")

    # H-plane (phi closest to pi/2)
    if len(ff.phi) > 1:
        h_idx = int(np.argmin(np.abs(ff.phi - np.pi / 2)))
        G_h = G_freq[:, h_idx]
        G_h_dB = 10.0 * np.log10(np.maximum(G_h, 1e-30))
        G_h_plot = np.maximum(G_h_dB, np.max(G_h_dB) - 40) - (np.max(G_h_dB) - 40)
        ax1.plot(ff.theta, G_h_plot, label="H-plane", linestyle="--")

    freq_ghz = ff.freqs[freq_idx] / 1e9
    ax1.set_title(f"Pattern ({freq_ghz:.2f} GHz)")
    ax1.legend(loc="upper right", fontsize=8)

    # --- Panel 2: Gain vs theta (rectangular) ---
    ax2 = fig.add_subplot(1, n_panels, 2)
    theta_deg = np.degrees(ff.theta)
    ax2.plot(theta_deg, G_e_dB, label="E-plane")
    if len(ff.phi) > 1:
        G_h_dB_rect = 10.0 * np.log10(np.maximum(G_freq[:, h_idx], 1e-30))
        ax2.plot(theta_deg, G_h_dB_rect, label="H-plane", linestyle="--")
    ax2.set_xlabel("Theta (degrees)")
    ax2.set_ylabel("Gain (dBi)")
    ax2.set_title("Gain vs Angle")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # --- Panel 3: S11 (if provided) ---
    if has_s11:
        ax3 = fig.add_subplot(1, n_panels, 3)
        s11_arr = np.asarray(s11)
        freqs_arr = np.asarray(freqs)
        if np.iscomplexobj(s11_arr):
            s11_dB = 20.0 * np.log10(np.maximum(np.abs(s11_arr), 1e-30))
        else:
            s11_dB = s11_arr.astype(np.float64)
        freqs_ghz = freqs_arr / 1e9
        ax3.plot(freqs_ghz, s11_dB, "b-")
        ax3.axhline(-10, color="r", linestyle="--", alpha=0.5, label="-10 dB")
        ax3.set_xlabel("Frequency (GHz)")
        ax3.set_ylabel("|S11| (dB)")
        ax3.set_title("Return Loss")
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig
