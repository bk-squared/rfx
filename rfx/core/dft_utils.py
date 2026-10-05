"""Shared DFT utilities used by probes and waveguide ports."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from rfx.measurement.dft import dft_window_weight, phase


def half_step_current_phase(freqs: jnp.ndarray, dt: float) -> jnp.ndarray:
    """Compatibility for legacy E-stamped current spectra, not scan records."""
    if dt == 0:
        return jnp.ones_like(freqs, dtype=jnp.complex64)
    return phase(0, freqs, dt, 'H') / phase(0, freqs, dt, 'E')


def port_dft_phase(step, freqs, dt, kind='E'):
    """Compatibility entry point for the physical measurement kernel."""
    return phase(jnp.squeeze(step), np.asarray(freqs).reshape(-1), dt, kind,
                 dtype=jnp.complex64)
