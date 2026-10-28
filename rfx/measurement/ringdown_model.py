"""Identified ringdown poles and residues, and the complex arithmetic they are read with.

Moved from ``rfx.ringdown``, which re-exports every name here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np


class RingdownPole(NamedTuple):
    """One identified pole.

    Signed frequency, loaded Q, amplitude decay rate, ``|lambda|`` per step,
    and its amplitude: the largest ``|residue|`` over the channels at the
    window's first sample, relative to that channel's RMS over the window.
    Channels include identification probes when requested, as well as ports.
    """

    f_hz: float
    q: float
    decay_per_s: float
    abs_lambda: float
    amplitude: float = math.nan


@dataclass(frozen=True)
class RingdownModel:
    """Poles and residues identified on a window of a multichannel record.

    ``s`` are the kept poles at the ORIGINAL step (1/s); ``c (K, C)`` their
    residues referred to sample ``n_ref`` (the window's first sample, so a
    fast pole's residue is never extrapolated back over the whole record).
    ``amplitude`` is each kept pole's largest ``|c| / RMS`` over the channels.
    ``s_growing`` are the poles the pencil discarded as growing, and
    ``growing_amplitude`` each one's size at the window's last sample,
    ``|r_k| |lambda_k|**(n_end - n_ref) / RMS`` (largest over the channels),
    from a residue fit made jointly with the kept poles.
    """

    s: np.ndarray
    c: np.ndarray
    n_ref: int
    dt: float
    freq_max: float
    guard: float | None
    factors: tuple
    D: int
    rank: int
    n_window: int
    n_decimated: int
    n_growing_discarded: int
    n_invalid: int
    n_guard_dropped: int
    singular_values: np.ndarray
    amplitude: np.ndarray = None
    s_growing: np.ndarray = None
    growing_amplitude: np.ndarray = None
    window_rms: np.ndarray = None

    @property
    def lam(self) -> np.ndarray:
        """Per-step eigenvalues ``exp(s dt)``."""
        return np.exp(self.s * self.dt)

    def predict(self, n) -> np.ndarray:
        """Model samples at absolute indices ``n`` -> ``(len(n), C)``."""
        n = np.asarray(n, dtype=np.float64)
        return np.exp(np.outer((n - self.n_ref) * self.dt, self.s)) @ self.c

    def _pole_arrays(self):
        f = self.s.imag / (2.0 * np.pi)
        alpha = -self.s.real
        lam = np.abs(self.lam)
        amp = (np.full(self.s.size, math.nan) if self.amplitude is None
               else np.asarray(self.amplitude, dtype=np.float64))
        return f, alpha, lam, amp

    @staticmethod
    def _pole_row(k, f, alpha, lam, amp) -> RingdownPole:
        q = (math.pi * abs(float(f[k])) / float(alpha[k])
             if alpha[k] != 0 else math.inf)
        return RingdownPole(float(f[k]), float(q), float(alpha[k]),
                            float(lam[k]), float(amp[k]))

    def pole(self, k: int) -> RingdownPole:
        """Kept pole ``k`` (its index in ``s``) as a :class:`RingdownPole`."""
        return self._pole_row(int(k), *self._pole_arrays())

    def poles(self) -> tuple:
        """Every kept pole as a :class:`RingdownPole`, largest amplitude first
        (by frequency when no amplitude is known)."""
        f, alpha, lam, amp = self._pole_arrays()
        order = (np.argsort(f, kind="stable") if np.all(np.isnan(amp))
                 else np.argsort(-np.nan_to_num(amp, nan=-1.0), kind="stable"))
        return tuple(self._pole_row(k, f, alpha, lam, amp) for k in order)


def _as_2d(series) -> tuple[np.ndarray, bool]:
    y = np.asarray(series)
    if y.ndim == 1:
        return y[:, None], True
    if y.ndim != 2:
        raise ValueError(f"series must be (n,) or (n, C), got shape {y.shape}")
    return y, False


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
