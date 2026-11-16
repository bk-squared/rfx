"""Reflection extraction from RIS simulation records."""

from __future__ import annotations

import numpy as np


def _extract_reflection(
    result,
    freq_range: tuple[float, float],
    n_freqs: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract S11 reflection from a simulation result.

    Tries Floquet S-params first — the only path here that returns a
    physically valid S11.

    WARNING: the time-domain FFT fallback below does NOT produce a valid
    S11. It FFTs the raw probe time series (total field, with no
    incident-field reference subtraction) and peak-normalizes the
    magnitude. This violates the repo rule "never FFT-of-probe for R(f)"
    and yields a spectrum shaped
    by the source, not a reflection coefficient. The fallback is kept
    only so RIS demos run end-to-end; treat its output as a qualitative
    placeholder, not a measured S-parameter.

    Returns
    -------
    s11 : (n_freqs,) complex array
    freqs : (n_freqs,) float array
    """
    # If the result has S-parameters from ports (rare for Floquet-only),
    # use them.
    if result.s_params is not None and result.freqs is not None:
        s11 = np.asarray(result.s_params)[0, 0, :]
        freqs = np.asarray(result.freqs)
        # Filter to freq_range
        mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
        if np.sum(mask) >= 2:
            return s11[mask], freqs[mask]

    # Fallback: FFT of probe time series
    ts = np.asarray(result.time_series).ravel()
    if result.dt is not None:
        dt = result.dt
    else:
        # Estimate dt from the simulation parameters
        dt = 1.0 / (freq_range[1] * 40)  # rough estimate

    n = len(ts)
    freqs_fft = np.fft.rfftfreq(n, d=dt)
    spectrum = np.fft.rfft(ts)

    # Interpolate to desired frequency points
    target_freqs = np.linspace(freq_range[0], freq_range[1], n_freqs)
    s11 = np.interp(target_freqs, freqs_fft, spectrum, left=0, right=0)

    # Normalize by peak
    peak = np.max(np.abs(s11))
    if peak > 0:
        s11 = s11 / peak

    return s11, target_freqs
