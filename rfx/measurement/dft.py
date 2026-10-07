"""One physical-leapfrog DFT for streaming samples and stored records.

Slot n holds E at (n+1) dt and H at (n+1/2) dt.

Concrete frequencies and dt: the phase is exact integer arithmetic. The
turns per step, f dt modulo one, and the stamp offset are prepared on the
host in float64 as 64-bit fixed-point numbers (two uint32 words per bin);
the scan forms n * turns + offset modulo one in wrapping uint32 arithmetic
and takes one cosine and sine of the result. Its error does not grow with
the step index (pre-declaration 6.2 / 7.3; measured <= 2e-7 rad in float32
at every step up to 2**31). Nothing is tabulated and nothing is cached, so
a compiled loop holds two words per bin and a sweep over dt or frequencies
accumulates neither memory nor compilations.

Traced frequencies or dt (a mesh or a bin list as a design variable): the
phase is evaluated in the trace, exp(-j 2 pi f (n + offset) dt), with the
same stamps and differentiable in both. Its float32 phase error grows with
the step index, as the arithmetic before S2 M2 did (known difference).
"""
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from .plan import Channel, TimeBase

_BLOCK = 256  # steps per matrix product in the stored-record replay


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


def _traced(*values):
    return any(isinstance(v, jax.core.Tracer) for v in values)


def _kind(kind):
    kind = kind.kind if isinstance(kind, Channel) else kind
    if kind not in ('E', 'H'):
        raise ValueError('DFT channel kind must be E or H')
    return kind


def _offset(kind, time_shift):
    clock = TimeBase(0.0, 0)
    return (clock.e_offset if kind == 'E' else clock.h_offset) + time_shift


def _fixed(turns):
    """Turns modulo one as 64-bit fixed point: (high, low) uint32 words."""
    scaled = np.mod(np.asarray(turns, np.float64), 1.0) * 2.0 ** 32  # exact: a power of two
    high = np.floor(scaled)
    low = np.floor((scaled - high) * 2.0 ** 32)
    return np.stack([np.mod(high, 2.0 ** 32), low]).astype(np.uint32)


def _words(freqs, dt, offset):
    """Host constants of one channel: per-step turns and stamp offset, (4, nf) uint32."""
    # The offset multiplies the whole product: reducing f dt modulo one
    # first would drop offset * floor(f dt) turns (half a turn for H).
    turns = freqs * dt
    return np.concatenate([_fixed(turns), _fixed(turns * offset)])


def _high_product(a, b):
    """High 32 bits of the 64-bit product of two uint32 arrays."""
    a0, a1, b0, b1 = a & 0xFFFF, a >> 16, b & 0xFFFF, b >> 16
    low, cross1, cross2 = a0 * b0, a0 * b1, a1 * b0
    middle = (low >> 16) + (cross1 & 0xFFFF) + (cross2 & 0xFFFF)
    return a1 * b1 + (cross1 >> 16) + (cross2 >> 16) + (middle >> 16)


def _unit(step, words, dtype):
    """exp(-j 2 pi (step * turns + offset)), the phase in 64-bit fixed point."""
    n = jnp.asarray(step, dtype=jnp.int32)[..., None]
    step_high, step_low, offset_high, offset_low = (jnp.asarray(w, jnp.uint32) for w in words)
    # Negative steps (low-level callers sampling the initial state):
    # n * turns = -(m * turns + turns) with m = -(n + 1) >= 0.
    negative = n < 0
    m = jnp.where(negative, -(n + 1), n).astype(jnp.uint32)
    low = m * step_low
    high = m * step_high + _high_product(m, step_low)
    low_n = low + step_low
    high_n = high + step_high + (low_n < low).astype(jnp.uint32)
    low_n, high_n = ~low_n + jnp.uint32(1), ~high_n + (low_n == 0).astype(jnp.uint32)
    low, high = jnp.where(negative, low_n, low), jnp.where(negative, high_n, high)
    total_low = low + offset_low
    total = high + offset_high + (total_low < low).astype(jnp.uint32)
    real = jnp.finfo(dtype).dtype
    if real == jnp.float64:
        angle = (2.0 * np.pi) * (total.astype(real) * 2.0 ** -32 + total_low.astype(real) * 2.0 ** -64)
        return (jnp.cos(angle) - 1j * jnp.sin(angle)).astype(dtype)
    # float32: the quarter turn from the top two bits, the remainder in
    # [0, pi/2) so its cosine and sine keep full relative precision.
    quarter = total >> 30
    angle = (total & jnp.uint32(0x3FFFFFFF)).astype(real) * real.type(np.pi / 2 * 2.0 ** -30)
    c, s = jnp.cos(angle), jnp.sin(angle)
    cosine = jnp.select([quarter == 0, quarter == 1, quarter == 2], [c, -s, -c], s)
    sine = jnp.select([quarter == 0, quarter == 1, quarter == 2], [s, c, -s], -c)
    return (cosine - 1j * sine).astype(dtype)


def _traced_unit(step, freqs, dt, offset, dtype):
    real = jnp.finfo(dtype).dtype
    time = (jnp.asarray(step, dtype=real) + jnp.asarray(offset, dtype=real)) * jnp.asarray(dt, dtype=real)
    angle = 2.0 * jnp.pi * jnp.ravel(jnp.asarray(freqs)).astype(real) * time[..., None]
    return jnp.exp(-1j * angle).astype(dtype)


def phase(step, freqs, dt, kind='E', *, dtype=None, time_shift=0.):
    """DFT weight including dt, with a single kind-dependent time offset.

    ``time_shift`` is in steps; it supports explicitly staged channels and
    clock mutations. No caller applies an additional H half-step factor.
    Concrete ``freqs`` and ``dt`` use the fixed-point phase; a traced one is
    evaluated in the trace (module docstring).
    """
    kind = _kind(kind)
    dtype = np.dtype(dtype or (np.complex128 if jax.config.x64_enabled else np.complex64))
    offset = _offset(kind, time_shift)
    real = jnp.finfo(dtype).dtype
    if _traced(freqs, dt, time_shift):
        return _traced_unit(step, freqs, dt, offset, dtype) * jnp.asarray(dt, dtype=real)
    frequencies = np.asarray(freqs, dtype=np.float64).ravel()
    return _unit(step, _words(frequencies, float(dt), offset), dtype) * jnp.asarray(dt, dtype=real)


def accumulate(acc, sample, step, freqs, dt, kind='E', *, total_steps=1,
               window='rect', alpha=.5, window_step=None, time_shift=0.):
    """Add one sampled channel; field dimensions follow the frequency axis."""
    weight = phase(step, freqs, dt, kind, dtype=acc.dtype, time_shift=time_shift)
    weight *= dft_window_weight(step if window_step is None else window_step,
                                total_steps, window, alpha)
    shape = weight.shape + (1,) * jnp.ndim(sample)
    return (acc + jnp.asarray(sample)[None, ...] * weight.reshape(shape)).astype(acc.dtype)


def transform(records, freqs, dt, kind='E', *, n_valid=None,
              window='rect', alpha=.5, time_shift=0.):
    """Replay stored records with the running kernel's stamps and weights.

    Row n of ``records`` is slot n. Samples, ``n_valid``, and (when traced)
    ``freqs`` and ``dt`` are array arguments of one compiled replay per
    record shape; nothing is specialised on their values.
    """
    records = jnp.asarray(records)
    offset = float(_offset(_kind(kind), time_shift))
    dtype = np.dtype(jnp.result_type(records.dtype, jnp.complex64))
    limit = len(records) if n_valid is None else n_valid
    if _traced(freqs, dt):
        return _replay(records, jnp.ravel(jnp.asarray(freqs)), dt, limit,
                       traced=True, offset=offset, window=window, alpha=alpha)
    frequencies = np.asarray(freqs, dtype=np.float64).ravel()
    return _replay(records, jnp.asarray(_words(frequencies, float(dt), offset)),
                   jnp.asarray(dt, dtype=jnp.finfo(dtype).dtype), limit,
                   traced=False, offset=offset, window=window, alpha=alpha)


@partial(jax.jit, static_argnames=('traced', 'offset', 'window', 'alpha'))
def _replay(records, spectrum, dt, limit, *, traced, offset, window, alpha):
    """``spectrum`` is the channel's fixed-point words, or the bins when ``traced``.

    The record is summed in blocks of ``_BLOCK`` steps, one matrix product
    per block, blocks added in order. The weights are the streaming
    kernel's; the order of additions inside a block is the product's, so a
    channel read both ways agrees to the accumulated-quantity bar (J7-1,
    1e-4 of peak), not bit for bit.
    """
    dtype = jnp.result_type(records.dtype, jnp.complex64)
    real = jnp.finfo(dtype).dtype
    length = records.shape[0]
    blocks = -(-length // _BLOCK)
    flat = jnp.pad(records.reshape(length, -1), ((0, blocks * _BLOCK - length), (0, 0)))
    steps = jnp.arange(blocks * _BLOCK).reshape(blocks, _BLOCK)
    initial = jnp.zeros((spectrum.shape[-1], flat.shape[1]), dtype=dtype)

    def body(acc, item):
        n, block = item
        block = jnp.where(((n < limit) & (n < length))[:, None], block, 0)
        if traced:
            unit = _traced_unit(n, spectrum, dt, offset, dtype)
        else:
            unit = _unit(n, spectrum, dtype)
        weight = unit * jnp.asarray(dt, dtype=real)
        weight *= jnp.broadcast_to(dft_window_weight(n, length, window, alpha), n.shape)[:, None]
        return acc + jnp.einsum('nf,nm->fm', weight, block.astype(dtype),
                                precision=jax.lax.Precision.HIGHEST), None

    total = jax.lax.scan(body, initial, (steps, flat.reshape(blocks, _BLOCK, -1)))[0]
    return total.reshape(spectrum.shape[-1], *records.shape[1:])
