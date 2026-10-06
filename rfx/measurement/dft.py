"""One physical-leapfrog DFT for streaming samples and stored records.

Slot n holds E at (n+1) dt and H at (n+1/2) dt.

Concrete frequencies and dt: the phase comes from host float64 tables, split
by the digits of the step index in base 64, so its float32 error does not
grow with the record (pre-declaration 6.2 / 7.3). Four digits (256 rows per
bin) are exact tables up to 64**4 = 16.8e6 steps; the part of the index
above that multiplies one more host-reduced factor in the trace, whose
float32 error grows only with that quotient (at most 127). A stored record
builds only the digits its length needs. Only the most recent table set is
kept, and stored-record replay receives it as an array argument, so a sweep
over dt or frequencies accumulates neither tables nor compilations.

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

_BASE = 64
_DIGITS = 4  # exact tables below 64**4 steps; see _unit for the rest
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


_recent = {}  # the most recent table set only: {'key': ..., 'tables': ...}


def _tables(freqs, dt, dtype, digits=_DIGITS):
    """Host table (digits, base, nf): exp(-j 2 pi f dt r base**d).

    The turns f dt are reduced modulo one before every scaling (a power of
    two, so exact), which keeps the float64 argument small at every digit.
    """
    key = (freqs.tobytes(), dt, dtype.str, digits)
    if _recent.get('key') != key:
        turns = np.fmod(freqs * dt, 1.0)
        rows = np.arange(_BASE, dtype=np.float64)[:, None]
        table = np.empty((digits, _BASE, freqs.size), dtype=dtype)
        angle, part = np.empty((2, _BASE, freqs.size))  # two work arrays, no other temporaries
        for digit in range(digits):
            np.multiply(rows, turns, out=angle)
            np.fmod(angle, 1.0, out=angle)
            angle *= -2.0 * np.pi
            table[digit].real = np.cos(angle, out=part)
            table[digit].imag = np.sin(angle, out=part)
            turns = np.fmod(turns * _BASE, 1.0)
        _recent.clear()
        _recent.update(key=key, tables=table)
    return _recent['tables']


def _fractional(freqs, dt, offset, dtype, digits=_DIGITS):
    """Stamp factors for steps n >= 0 and for -(n+1), and the factor of one
    unit of the index above the tables (base**digits steps)."""
    turns = freqs * dt
    above = np.fmod(np.fmod(turns, 1.0) * float(_BASE) ** digits, 1.0)
    return np.stack([np.exp(-2j * np.pi * turns * offset),
                     np.exp(-2j * np.pi * turns * (offset - 1)),
                     np.exp(-2j * np.pi * above)]).astype(dtype)


def _unit(step, tables, fractional):
    """exp(-j 2 pi f dt (step + offset)) from the digit tables."""
    n = jnp.asarray(step, dtype=jnp.int32)
    # Negative steps support low-level callers sampling the initial state.
    # -(n+1) also avoids overflow at the int32 lower bound.
    negative = n < 0
    index = jnp.where(negative, -(n + 1), n)
    unit = tables[0][index % _BASE]
    for digit in range(1, tables.shape[0]):
        index = index // _BASE
        unit = unit * tables[digit][index % _BASE]
    above = (index // _BASE)[..., None]
    unit = unit * jnp.where(above > 0, fractional[2] ** above, 1)
    return jnp.where(negative[..., None], jnp.conj(unit) * fractional[1],
                     unit * fractional[0])


def _traced_unit(step, freqs, dt, offset, dtype):
    real = jnp.finfo(dtype).dtype
    time = (jnp.asarray(step, dtype=real) + jnp.asarray(offset, dtype=real)) * jnp.asarray(dt, dtype=real)
    angle = 2.0 * jnp.pi * jnp.ravel(jnp.asarray(freqs)).astype(real) * time[..., None]
    return jnp.exp(-1j * angle).astype(dtype)


def phase(step, freqs, dt, kind='E', *, dtype=None, time_shift=0.):
    """DFT weight including dt, with a single kind-dependent time offset.

    ``time_shift`` is in steps; it supports explicitly staged channels and
    clock mutations. No caller applies an additional H half-step factor.
    Concrete ``freqs`` and ``dt`` use the host tables; a traced one is
    evaluated in the trace (module docstring).
    """
    kind = _kind(kind)
    dtype = np.dtype(dtype or (np.complex128 if jax.config.x64_enabled else np.complex64))
    offset = _offset(kind, time_shift)
    real = jnp.finfo(dtype).dtype
    if _traced(freqs, dt, time_shift):
        return _traced_unit(step, freqs, dt, offset, dtype) * jnp.asarray(dt, dtype=real)
    frequencies = np.asarray(freqs, dtype=np.float64).ravel()
    unit = _unit(step, jnp.asarray(_tables(frequencies, float(dt), dtype)),
                 jnp.asarray(_fractional(frequencies, float(dt), offset, dtype)))
    return unit * jnp.asarray(dt, dtype=real)


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
        return _replay(records, jnp.ravel(jnp.asarray(freqs)), None, dt, limit,
                       offset=offset, window=window, alpha=alpha)
    frequencies = np.asarray(freqs, dtype=np.float64).ravel()
    digits = 1
    while _BASE ** digits < len(records) and digits < _DIGITS:
        digits += 1
    return _replay(records, jnp.asarray(_tables(frequencies, float(dt), dtype, digits)),
                   jnp.asarray(_fractional(frequencies, float(dt), offset, dtype, digits)),
                   jnp.asarray(dt, dtype=jnp.finfo(dtype).dtype), limit,
                   offset=offset, window=window, alpha=alpha)


@partial(jax.jit, static_argnames=('offset', 'window', 'alpha'))
def _replay(records, spectrum, fractional, dt, limit, *, offset, window, alpha):
    """``spectrum`` is the table set, or the bins when ``fractional`` is None.

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
        if fractional is None:
            unit = _traced_unit(n, spectrum, dt, offset, dtype)
        else:
            unit = _unit(n, spectrum, fractional)
        weight = unit * jnp.asarray(dt, dtype=real)
        weight *= jnp.broadcast_to(dft_window_weight(n, length, window, alpha), n.shape)[:, None]
        return acc + jnp.einsum('nf,nm->fm', weight, block.astype(dtype),
                                precision=jax.lax.Precision.HIGHEST), None

    total = jax.lax.scan(body, initial, (steps, flat.reshape(blocks, _BLOCK, -1)))[0]
    return total.reshape(spectrum.shape[-1], *records.shape[1:])
