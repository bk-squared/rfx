"""NumPy record DFT and ringdown tail spectra."""

from __future__ import annotations

import numpy as np
from rfx.measurement.ringdown_model import RingdownModel, _as_2d, _cmul


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
