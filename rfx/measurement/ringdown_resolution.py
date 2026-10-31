"""Pole classes and in-window shares at the identification's rate resolution."""

import numpy as np

from rfx.measurement.ringdown_tail import plain_dft


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
