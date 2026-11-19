"""Column drive policy for S calculators, independent of run declarations."""

import jax.numpy as jnp

from rfx.core.jax_utils import is_tracer


def calculator_drive(selected: bool, *, amplitude=None, excite=True):
    """Enable the selected column, retaining every existing nonzero level.

    Boolean ports use the returned excitation flag with their own waveform
    (or their calculator's default). Their declared ``excite`` flag describes
    a run record, not the calculator's drive. Waveguide ports pass amplitude:
    only a zero declaration needs the calculator's fallback level of 1.0.
    """
    if amplitude is None:
        return selected
    if not selected:
        return 0.0
    if is_tracer(amplitude):
        return jnp.where(amplitude == 0, 1.0, amplitude)
    return 1.0 if amplitude == 0 else amplitude
