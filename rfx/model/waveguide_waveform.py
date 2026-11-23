"""Waveguide declaration frequency and incident-table waveform admission."""

from contextlib import nullcontext

import jax
import jax.numpy as jnp
import numpy as np

from rfx.core.jax_utils import is_tracer

DG_REFUSAL = (
    "waveform='differentiated_gaussian' is refused on waveguide ports: its "
    "band-limited incident table is built with a zero-phase filter whose "
    "precursor wraps circularly to the end of the record, so the port "
    "re-injects at the record end (an empty matched WR-90 guide read S21 off "
    "by up to 11 dB). Use waveform='modulated_gaussian' (the default)."
    ""
)


def waveguide_waveform_error(waveform):
    """Return the declaration and table builder's shared refusal message."""
    if waveform == "differentiated_gaussian":
        return DG_REFUSAL
    return f"waveform must be 'modulated_gaussian', got {waveform!r}"


def admit_waveguide_freqs(freqs):
    """Convert and validate concrete frequencies without staging host checks."""
    with nullcontext() if is_tracer(freqs) else jax.ensure_compile_time_eval():
        freqs_arr = jnp.asarray(freqs)
        if freqs_arr.ndim != 1 or freqs_arr.size == 0:
            raise ValueError("freqs must be a non-empty 1-D array")
        freqs_np = np.asarray(freqs_arr, dtype=float)
        if not np.all(np.isfinite(freqs_np)):
            raise ValueError("freqs must contain only finite values")
        if np.any(freqs_np <= 0):
            raise ValueError("freqs must contain only positive values")
        return freqs_arr
