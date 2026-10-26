"""Array conventions shared by ringdown identification and DFTs."""

import numpy as np


def _as_2d(series) -> tuple[np.ndarray, bool]:
    y = np.asarray(series)
    if y.ndim == 1:
        return y[:, None], True
    if y.ndim != 2:
        raise ValueError(f"series must be (n,) or (n, C), got shape {y.shape}")
    return y, False


def _given_spectra(arr, nf: int, n_ch: int, name: str) -> np.ndarray:
    """A caller-supplied ``(nf,)`` or ``(nf, C)`` spectrum as ``(nf, C)``, or raise."""
    a = np.asarray(arr)
    if a.ndim == 1:
        a = a[:, None]
    if a.shape != (nf, n_ch):
        raise ValueError(f"{name} has shape {np.shape(arr)}; expected ({nf}, {n_ch}) "
                         f"or ({nf},) for a single channel")
    return a


def _cmul(a, b) -> np.ndarray:
    """``a * b`` for complex arrays (numpy broadcasting), the same bits at any address.

    numpy 1.26 on arm64 multiplies complex arrays in two loops that round
    differently (one fuses a multiply and an add), and which loop an element
    takes depends on where the arrays sit in memory: the same product came out
    two ways in one process (2 distinct results over 200 allocations; numpy
    2.4 gave 1), so a completion computed twice from one model could differ in
    its last digit. Built from real products, each rounded once, the result
    does not depend on the address.
    """
    a = np.asarray(a, dtype=np.complex128)
    b = np.asarray(b, dtype=np.complex128)
    out = np.empty(np.broadcast_shapes(a.shape, b.shape), dtype=np.complex128)
    out.real = a.real * b.real - a.imag * b.imag
    out.imag = a.real * b.imag + a.imag * b.real
    return out


