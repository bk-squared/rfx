"""One physical-leapfrog DFT for streaming samples and stored records.

Slot n holds E at (n+1) dt and H at (n+1/2) dt. Phase tables are
prepared in host float64; float32 scans only index and multiply factors.
The high index is itself split for the full nonnegative int32 clock range,
so table storage is bounded independently of record length.
"""
from functools import lru_cache, partial

import jax
import jax.numpy as jnp
import numpy as np

from .plan import Channel, TimeBase


def dft_window_weight(step: int, total_steps: int, window: str, alpha: float) -> jnp.ndarray:
    """Streaming DFT window weight for a given timestep.

    Parameters
    ----------
    step : int
        Current timestep index.
    total_steps : int
        Total number of timesteps in the simulation.
    window : str
        Window type: ``"rect"``, ``"hann"``, or ``"tukey"``.
    alpha : float
        Shape parameter for the Tukey window (ignored for others).

    Returns
    -------
    jnp.ndarray
        Scalar weight in [0, 1].
    """
    if total_steps <= 1 or window == "rect":
        return jnp.asarray(1.0, dtype=jnp.float32)

    n = jnp.asarray(step, dtype=jnp.float32)
    N = jnp.asarray(total_steps - 1, dtype=jnp.float32)
    x = n / jnp.maximum(N, 1.0)

    if window == "hann":
        return 0.5 * (1.0 - jnp.cos(2.0 * jnp.pi * x))

    if window == "tukey":
        a = float(alpha)
        if a <= 0.0:
            return jnp.asarray(1.0, dtype=jnp.float32)
        if a >= 1.0:
            return 0.5 * (1.0 - jnp.cos(2.0 * jnp.pi * x))
        left = 0.5 * (1.0 + jnp.cos(jnp.pi * (2.0 * x / a - 1.0)))
        right = 0.5 * (1.0 + jnp.cos(jnp.pi * (2.0 * x / a - 2.0 / a + 1.0)))
        return jnp.where(
            x < a / 2.0,
            left,
            jnp.where(x <= 1.0 - a / 2.0, 1.0, right),
        )

    raise ValueError(f"dft_window must be 'rect', 'hann', or 'tukey', got {window!r}")


@lru_cache(maxsize=32)
def _tables(frequencies, dt, dtype):
    phi = np.remainder(2 * np.pi * np.asarray(frequencies, dtype=np.float64) * dt,
                       2 * np.pi)
    # n = n_lo + 4096*n_hi; factor n_hi once more to cover int32.
    tables = tuple(np.exp(-1j * (np.arange(size, dtype=np.float64)[:, None]
                                 * stride * phi)).astype(dtype)
                   for stride, size in ((1, 4096), (4096, 4096), (16777216, 128)))
    return tables


def phase(step, freqs, dt, kind='E', *, dtype=None, time_shift=0.):
    """DFT weight including dt, with a single kind-dependent time offset.

    Frequencies and dt are run metadata, concrete when the scan is staged.
    ``time_shift`` is in steps; it supports explicitly staged channels and
    clock mutations. No caller applies an additional H half-step factor.
    """
    if isinstance(kind, Channel):
        kind = kind.kind
    if kind not in ('E', 'H'):
        raise ValueError('DFT channel kind must be E or H')
    dtype = np.dtype(dtype or (np.complex128 if jax.config.x64_enabled else np.complex64))
    clock = TimeBase(float(dt), 0)
    offset = (clock.e_offset if kind == 'E' else clock.h_offset) + time_shift
    frequencies = tuple(np.asarray(freqs).ravel())
    low, high, top = _tables(frequencies, float(dt), dtype.str)
    fractional = np.exp(-2j * np.pi * np.asarray(frequencies, dtype=np.float64)
                        * float(dt) * offset).astype(dtype)
    negative_fractional = np.exp(-2j * np.pi * np.asarray(frequencies, dtype=np.float64)
                                 * float(dt) * (offset-1)).astype(dtype)
    n = jnp.asarray(step, dtype=jnp.int32)
    # Negative steps support low-level callers sampling the initial state.
    # -(n+1) also avoids overflow at the int32 lower bound.
    negative = n < 0
    index = jnp.where(negative, -(n+1), n)
    unit = (jnp.asarray(low)[index % 4096] * jnp.asarray(high)[(index // 4096) % 4096]
            * jnp.asarray(top)[index // 16777216])
    unit = jnp.where(negative[..., None], jnp.conj(unit)*jnp.asarray(negative_fractional),
                     unit*jnp.asarray(fractional))
    return unit * jnp.asarray(dt, dtype=jnp.finfo(dtype).dtype)


def accumulate(acc, sample, step, freqs, dt, kind='E', *, total_steps=1,
               window='rect', alpha=.5, window_step=None, time_shift=0.):
    """Add one sampled channel; field dimensions follow the frequency axis."""
    weight = phase(step, freqs, dt, kind, dtype=acc.dtype, time_shift=time_shift)
    weight *= dft_window_weight(step if window_step is None else window_step,
                                total_steps, window, alpha)
    shape = weight.shape + (1,) * jnp.ndim(sample)
    return (acc + jnp.asarray(sample)[None, ...] * weight.reshape(shape)).astype(acc.dtype)


def transform(records, freqs, dt, kind='E', *, first_step=0, n_valid=None,
              window='rect', alpha=.5, time_shift=0., window_step_offset=0):
    """Replay stored records with the same summation as the running kernel.

    ``window_step_offset`` preserves an existing record's window indexing;
    it changes no sample time. Run metadata is concrete, samples are traced.
    """
    frequencies = tuple(np.asarray(freqs).ravel())
    kind = kind.kind if isinstance(kind, Channel) else kind
    return _replay(jnp.asarray(records), frequencies, float(dt), kind, first_step,
                   len(records) if n_valid is None else n_valid,
                   window, alpha, time_shift, window_step_offset)


@partial(jax.jit, static_argnames=('freqs', 'dt', 'kind', 'first_step', 'window',
                                   'alpha', 'time_shift', 'window_step_offset'))
def _replay(records, freqs, dt, kind, first_step, limit, window, alpha,
            time_shift, window_step_offset):
    dtype = jnp.result_type(records.dtype, jnp.complex64)
    initial = jnp.zeros((len(freqs), *records.shape[1:]), dtype=dtype)
    def body(acc, item):
        n, sample = item
        sample = jnp.where(n < limit, sample, 0)
        return accumulate(acc, sample, n+first_step, freqs, dt, kind,
                          total_steps=len(records), window=window, alpha=alpha,
                          window_step=n+window_step_offset,
                          time_shift=time_shift), None
    return jax.lax.scan(body, initial, (jnp.arange(len(records)), records))[0]
