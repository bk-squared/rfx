"""Shared DFT utilities used by probes and waveguide ports."""

from __future__ import annotations

import jax.numpy as jnp

from rfx.measurement.dft import dft_window_weight, half_step_current_phase, phase


def port_dft_phase(step, freqs, dt, kind='E', *, dtype=jnp.complex64):
    """Port entry point for the physical measurement kernel.

    ``dtype`` is the accumulator's dtype: complex64 on a float32 run,
    complex128 on a float64 run, so the weights are as exact as the sum.
    """
    return phase(jnp.squeeze(step), freqs, dt, kind, dtype=dtype)
