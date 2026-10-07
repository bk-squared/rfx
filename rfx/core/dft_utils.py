"""Shared DFT utilities used by probes and waveguide ports."""

from __future__ import annotations

import jax.numpy as jnp

from rfx.measurement.dft import dft_window_weight, phase


def half_step_current_phase(freqs: jnp.ndarray, dt: float) -> jnp.ndarray:
    """exp(+j pi f dt): turns an E-stamped current spectrum into an H-stamped one.

    For spectra supplied already stamped at E's time (legacy callers, the
    ring-down's slot-relative completion). Scan records never need it: the
    kernel stamps H itself.
    """
    return jnp.exp(1j * jnp.pi * jnp.asarray(freqs) * dt)


def port_dft_phase(step, freqs, dt, kind='E', *, dtype=jnp.complex64):
    """Port entry point for the physical measurement kernel.

    ``dtype`` is the accumulator's dtype: complex64 on a float32 run,
    complex128 on a float64 run, so the weights are as exact as the sum.
    """
    return phase(jnp.squeeze(step), freqs, dt, kind, dtype=dtype)
