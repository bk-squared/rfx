"""NumPy record DFT and ringdown tail spectra."""

from __future__ import annotations

import numpy as np
from rfx.ringdown import RingdownModel, _as_2d, _cmul


def plain_dft(series, dt, freqs, *, chunk: int = 4096) -> np.ndarray:
    """``dt * sum_n y_n exp(-j 2 pi f n dt)`` over the whole ``series``.

    ``series`` is ``(n,)`` or ``(n, C)``; returns ``(nf,)`` or ``(nf, C)``,
    complex128, float64 kernel.
    """
    Y, one_d = _as_2d(series)
    freqs = np.asarray(freqs, dtype=np.float64)
    w = -2.0 * np.pi * freqs * float(dt)
    acc = np.zeros((freqs.size, Y.shape[1]), dtype=np.complex128)
    for n0 in range(0, Y.shape[0], chunk):
        n1 = min(n0 + chunk, Y.shape[0])
        nn = np.arange(n0, n1, dtype=np.float64)
        acc += np.exp(1j * np.outer(w, nn)) @ Y[n0:n1].astype(np.complex128)
    out = acc * float(dt)
    return out[:, 0] if one_d else out


def geometric_term_dft(s, coefficient, count, dt, bins, *, end_referenced=False):
    """A fitted term's window DFT, summed from its coefficient's reference.

    expm1 retains near-bin terms. Growing discarded poles run backwards
    from the last sample so all their real exponents are non-positive
    (witness-cost design Addendum 3).
    """
    phase = -2.0 * np.pi * np.asarray(bins, dtype=np.float64) * float(dt)
    u = s * float(dt) + 1j * phase
    if end_referenced:
        u = -u
    with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
        geometric = np.expm1(count * u) / np.expm1(u)
    geometric = np.where(u == 0, count, geometric)
    if end_referenced:
        geometric *= np.exp(1j * phase * (count - 1))
    return float(dt) * geometric[:, None] * coefficient


def tail_dft(model: RingdownModel, n_last: int, freqs) -> np.ndarray:
    """The unrecorded part of each spectrum: samples ``n_last + 1 .. inf``.

    ``dt * sum_k c_k lambda_k**(N+1-n_ref) z**(N+1) / (1 - lambda_k z)`` with
    ``N = n_last`` and ``1 - lambda_k z = -expm1(s_k dt - j 2 pi f dt)``.
    Returns ``(nf, C)``.
    """
    freqs = np.asarray(freqs, dtype=np.float64)
    s = np.asarray(model.s, dtype=np.complex128)
    c = np.asarray(model.c, dtype=np.complex128)
    dt = float(model.dt)
    if s.size == 0:
        return np.zeros((freqs.size, c.shape[1]), dtype=np.complex128)
    n1 = int(n_last) + 1
    wdt = 2.0 * np.pi * freqs * dt                                  # (nf,)
    amp = _cmul(np.exp(s * dt * (n1 - int(model.n_ref)))[:, None], c)   # (K, C)
    zn1 = np.exp(-1j * wdt * n1)                                    # (nf,)
    one_minus = -np.expm1(s[None, :] * dt - 1j * wdt[:, None])      # (nf, K)
    return dt * ((zn1[:, None] / one_minus) @ amp)


def _pole_tails(model: RingdownModel, n_last: int, freqs) -> np.ndarray:
    """Each kept pole's term of :func:`tail_dft`: ``(nf, K, C)``, K in ``model.s`` order."""
    freqs = np.asarray(freqs, dtype=np.float64)
    s = np.asarray(model.s, dtype=np.complex128)
    c = np.asarray(model.c, dtype=np.complex128)
    dt = float(model.dt)
    if s.size == 0:
        return np.zeros((freqs.size, 0, c.shape[1]), dtype=np.complex128)
    n1 = int(n_last) + 1
    wdt = 2.0 * np.pi * freqs * dt
    amp = _cmul(np.exp(s * dt * (n1 - int(model.n_ref)))[:, None], c)   # (K, C)
    zn1 = np.exp(-1j * wdt * n1)
    one_minus = -np.expm1(s[None, :] * dt - 1j * wdt[:, None])          # (nf, K)
    return dt * _cmul((zn1[:, None] / one_minus)[:, :, None], amp[None, :, :])
