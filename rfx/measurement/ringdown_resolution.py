"""Pole classes and in-window shares at the identification's rate resolution."""

from __future__ import annotations

import numpy as np

from rfx.measurement.ringdown_model import RingdownPole
from rfx.measurement.ringdown_tail import plain_dft


def _zero_frequency(f_hz: float, record_s: float) -> bool:
    """``|f| < 1 / record``: the record cannot tell this pole's frequency from DC.

    The one definition the growing-pole exemption (:func:`growing_poles`) and
    the early stop's floor (:func:`_floor_pole`) share.
    """
    return abs(float(f_hz)) < 1.0 / float(record_s)


def _static_pole(p: RingdownPole, record_s: float, unit_tol: float) -> bool:
    """A static field: ``|f| < 1 / record`` and ``|1 - |lambda|| < unit_tol``.

    A field left behind in a closed box identifies as a pole the record cannot
    tell from DC (:func:`_zero_frequency`) on the unit circle to rounding
    (the Q 533 cavity). The one definition the growing-pole
    exemption (:func:`growing_poles`) and the early stop's floor
    (:func:`_floor_pole`) share; a decaying low-frequency mode is not one.
    """
    return (_zero_frequency(p.f_hz, record_s)
            and abs(p.abs_lambda - 1.0) < float(unit_tol))


def pole_classes(model, unit_tol=1e-6):
    """Return -1 (decaying), 0 (unresolved), +1 (growing), per original step.

    The symmetric edge is log(1 + unit_tol) / D, including equality in the
    resolved classes. Static exemptions remain the caller's responsibility.
    """
    edge = np.log1p(unit_tol) / model.D
    rate = model.s.real * model.dt
    return np.where(rate <= -edge, -1, np.where(rate >= edge, 1, 0))


def in_window_shares(model, indices, bins, denominator):
    """Absolute fitted-term DFT / caller denominator, shaped (bins, poles, C)."""
    time = np.arange(model.n_window) * model.dt
    values = np.zeros((len(bins), len(indices), model.c.shape[1]))
    for j, k in enumerate(indices):
        samples = np.exp(time * model.s[k])[:, None] * model.c[k]
        with np.errstate(divide='ignore', invalid='ignore'):
            values[:, j, :] = np.abs(plain_dft(samples, model.dt, bins)) / denominator
    return values


def completion_poles(model, n_record, unit_tol=1e-6):
    """Poles and a completion mask; unresolved non-static tails are omitted."""
    classes = pole_classes(model, unit_tol)
    keep = np.array([classes[k] != 0 or _static_pole(
        model.pole(k), n_record * model.dt, unit_tol)
        for k in range(len(model.s))], dtype=bool)
    return model.s, keep
