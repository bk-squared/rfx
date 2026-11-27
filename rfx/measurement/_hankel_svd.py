"""FFT Hankel products and a residual-certified, cost-limited partial SVD."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable

import numpy as np


def _partial_memory_bytes(shape, columns, b):
    """Charge spectra, channel FFT work, block QR work and the small SVD."""
    nd, channels = shape
    n = 1 << (nd - 1).bit_length()
    rows, width = nd - columns, channels * columns
    return 3 * 16 * (channels*n + 2*b*n + 3*(rows + width)*b + 4*b*b)


class _Hankel:
    """Side-by-side Hankel stack, without allocating either shifted matrix.

    Inputs are complex128 channels with the pencil's RMS normalization.
    Products accept column blocks; shift one applies Y1 instead of Y0.
    Channel work is sequential so FFT buffers do not scale with channels.
    """

    def __init__(self, y, columns):
        self.nd, self.channels = y.shape
        self.columns = columns
        self.rows = self.nd - columns
        self.width = self.channels * columns
        self.n = 1 << (self.nd - 1).bit_length()
        self.spectra = np.fft.fft(y, n=self.n, axis=0)
        self.phi = float(np.linalg.norm(np.max(np.abs(self.spectra), axis=0)))
        u = np.finfo(np.float64).eps / 2
        mu = 2*u
        eta = mu + 4*u/(1 - 4*u)*(np.sqrt(2) + mu)
        g = np.log2(self.n)*eta/(1 - np.log2(self.n)*eta)
        self.beta = 2*g + np.sqrt(2)*2*u/(1 - 2*u) + (self.channels - 1)*u

    def product(self, x, *, adjoint=False, shift=0):
        """Apply Y_shift or its conjugate transpose to a column block."""
        b = x.shape[1]
        if adjoint:
            out = np.empty((self.width, b), dtype=np.complex128)
            for ch in range(self.channels):
                transform = np.fft.fft(x[::-1].conj(), n=self.n, axis=0)
                transform *= self.spectra[:, ch, None]
                work = np.fft.ifft(transform, axis=0)
                start = self.rows - 1 + shift
                out[ch*self.columns:(ch + 1)*self.columns] = work[
                    start:start + self.columns].conj()
        else:
            out = np.zeros((self.rows, b), dtype=np.complex128)
            for ch in range(self.channels):
                block = x[ch*self.columns:(ch + 1)*self.columns]
                transform = np.fft.fft(block[::-1], n=self.n, axis=0)
                transform *= self.spectra[:, ch, None]
                work = np.fft.ifft(transform, axis=0)
                start = self.columns - 1 + shift
                out += work[start:start + self.rows]
        return out

    def product_cost(self, b):
        """Real operation count for one b-column FFT product."""
        return self.channels*b*(10*self.n*np.log2(self.n) + 6*self.n)


@dataclass
class _PartialSVD:
    """Certification result, including the cost-stop audit at return.

    A nonempty cause is unresolved: count for a prohibited cost step,
    block for a memory admission failure, or b>k for a prohibited doubling.
    Only the ringdown wrapper may decide fallback or raise a refusal.
    """

    q: int
    b: int
    counter: float
    full_cost: float
    cause: str
    operator: _Hankel | None = None
    u: np.ndarray | None = None
    s: np.ndarray | None = None
    vh: np.ndarray | None = None
    c: float | None = None


def _qr_cost(m, b):
    """Complex Householder QR cost expressed in real operations."""
    return 8*m*b*b - (8/3)*b**3


def _orthonormalize(block):
    """Reorthonormalize after every Hankel product in the power iteration."""
    return np.linalg.qr(block, mode="reduced")[0]


def _partial_svd(y, columns, sv_rel, fits: Callable[[int], bool]):
    """HMT block iteration with QR after every product and a two-sided rule.

    Every product, QR and small SVD is charged before it runs. The retained
    directions and first discarded direction must meet the FFT rounding
    bound, with at least ten discarded Ritz values present in the sketch.
    """
    nd, channels = y.shape
    rows, width = nd - columns, channels*columns
    k = min(rows, width)
    full_cost = 4*(6*max(rows, width)*k*k + 20*k**3)
    result = _PartialSVD(0, 32, 0, full_cost, "")

    def charge(cost):
        if result.counter + cost > full_cost:
            result.cause = "count"
            return False
        result.counter += cost
        return True

    n = 1 << (nd - 1).bit_length()
    if not charge(channels*5*n*np.log2(n)):
        return result
    if not fits(result.b):
        result.cause = "block"
        return result
    op = _Hankel(y, columns)
    result.operator = op
    while True:
        b = result.b
        if b > k:
            result.cause = "b>k"
            return result
        if not fits(b):
            result.cause = "block"
            return result
        if not charge(op.product_cost(b) + _qr_cost(rows, b)):
            return result
        rng = np.random.default_rng(0)
        omega = (rng.standard_normal((width, b))
                 + 1j*rng.standard_normal((width, b))) / np.sqrt(2)
        qblock = _orthonormalize(op.product(omega))
        del omega
        result.q = 0
        while True:
            if result.q >= 3:
                if not charge(op.product_cost(b) + 4*(6*width*b*b + 20*b**3)):
                    return result
                small = op.product(qblock, adjoint=True).conj().T
                ub, s, vh = np.linalg.svd(small, full_matrices=False)
                del small
                if not s[0] > 0:
                    raise ValueError("the stacked Hankel matrix of the window is zero")
                if np.count_nonzero(s <= sv_rel*s[0]) < 10:
                    if 2*b > k:
                        result.cause = "b>k"
                        return result
                    result.b = 2*b
                    del qblock, ub, s, vh
                    break
                rank = max(int(np.count_nonzero(s > sv_rel*s[0])), 1)
                j = min(rank + 1, b)
                if not charge(2*op.product_cost(j)):
                    return result
                u = qblock @ ub
                v = vh[:j].conj().T
                left = op.product(v) - u[:, :j]*s[:j]
                right = op.product(u[:, :j], adjoint=True) - v*s[:j]
                rho = np.sqrt(np.sum(np.abs(left)**2, axis=0)
                              + np.sum(np.abs(right)**2, axis=0))
                threshold = 2*np.sqrt(2)*op.beta*op.phi
                if np.max(rho) <= threshold:
                    result.u, result.s, result.vh = u, s, vh
                    result.c = threshold/(np.finfo(np.float64).eps*s[0])
                    return result
                del u, v, left, right, rho, ub, s, vh
            cost = 2*op.product_cost(b) + _qr_cost(width, b) + _qr_cost(rows, b)
            if not charge(cost):
                return result
            w = _orthonormalize(op.product(qblock, adjoint=True))
            qblock = _orthonormalize(op.product(w))
            del w
            result.q += 1
