"""Time-blocked QR and frozen first derivatives of a completion window."""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from rfx.measurement._block_plan import block_plan
from rfx._internal._precision import HIGHEST
from rfx.core.jax_utils import complex_solve_triangular

_BLOCK_TARGET_BYTES = 8 * 2**20


def _block_plan(length, width, dtype):
    """Balance the target's block count, capped at length // width (minimum one).

    Every real block has at least width rows when length permits; sizes differ
    by at most one. Shorter windows form one block. Read the target at call
    time; a jitted call keeps its traced plan.
    """
    return block_plan(length, width, _BLOCK_TARGET_BYTES, jnp.dtype(dtype).itemsize)


def _rows(s, dt, start, size, length, mask):
    """Reproduce _basis's window-relative working-precision times, including padding."""
    from rfx.ringdown._jax import working_dtypes

    rdt, _ = working_dtypes()
    indices = start + jnp.arange(size)
    t = indices.astype(rdt) * dt
    valid = indices < length
    basis = jnp.exp(t[:, None] * s[None, :]) * mask[None, :]
    basis = jnp.where(valid[:, None], basis, 0)
    return basis, t[:, None] * basis


def _tri(r, b, *, adjoint=False):
    return complex_solve_triangular(jnp.conj(r).T if adjoint else r, b, lower=adjoint)


def _norms(s, dt, mask, plan):
    size = max(stop - start for start, stop in plan)
    k = s.size

    def body(carry, window):
        start, stop = window
        b, tb = _rows(s, dt, start, size, stop, mask)
        return (carry[0] + jnp.sum(jnp.abs(b)**2, axis=0),
                carry[1] + jnp.sum(jnp.abs(tb)**2, axis=0)), None

    sums, _ = jax.lax.scan(body, (jnp.zeros(k, s.real.dtype), jnp.zeros(k, s.real.dtype)), jnp.asarray(plan).reshape(-1, 2))
    return tuple(jnp.where(mask > 0, jnp.sqrt(v), 1) for v in sums)


def _factor(y, s, dt, mask, plan):
    """Factor exactly the balanced real-row boundaries used by the products scan."""
    size = max(stop - start for start, stop in plan)
    length, channels = y.shape
    k = s.size
    width = 2 * k + channels
    nb, nt = _norms(s, dt, mask, plan)
    initial = jnp.zeros((width, width), s.dtype)
    unit = jnp.diag(1 - mask)
    initial = initial.at[:k, :k].set(unit)
    initial = initial.at[k:2*k, k:2*k].set(unit)
    padded = jnp.pad(y, ((0, max(start + size for start, stop in plan) - length), (0, 0)))

    def body(r, window):
        start, stop = window
        b, tb = _rows(s, dt, start, size, stop, mask)
        yw = jax.lax.dynamic_slice(padded, (start, 0), (size, channels))
        yw = jnp.where((start + jnp.arange(size) < stop)[:, None], yw, 0)
        rows = jnp.concatenate((b / nb, tb / nt, yw), axis=1)
        with jax.default_matmul_precision("highest"):
            r = jnp.linalg.qr(jnp.concatenate((r, rows)), mode="r")
        return r, None

    r, _ = jax.lax.scan(body, initial, jnp.asarray(plan).reshape(-1, 2))
    return r[:k, :k], r[:k, k:2*k], r[:k, 2*k:], r[k:2*k, k:2*k], r[k:2*k, 2*k:], nb, nt


def _products(y, s, dt, mask, nb, nt, plan):
    """Checkpoint each rebuilt basis block, including in transposed linear products."""
    length, channels = y.shape
    size = max(stop - start for start, stop in plan)
    padded = jnp.pad(y, ((0, max(start + size for start, stop in plan) - length), (0, 0)))

    @jax.checkpoint
    def body(carry, window):
        start, stop = window
        b, tb = _rows(s, dt, start, size, stop, mask)
        yw = jax.lax.dynamic_slice(padded, (start, 0), (size, channels))
        yw = jnp.where((start + jnp.arange(size) < stop)[:, None], yw, 0)
        return (carry[0] + jnp.matmul(jnp.conj(b / nb).T, yw, precision=HIGHEST),
                carry[1] + jnp.matmul(jnp.conj(tb / nt).T, yw, precision=HIGHEST)), None

    zero = jnp.zeros((s.size, channels), s.dtype)
    return jax.lax.scan(body, (zero, zero), jnp.asarray(plan).reshape(-1, 2))[0]


def _project(y, s, dt, mask, blocks, plan):
    r11, r12, _r13, r22, _r23, nb, nt = blocks
    bh, th = _products(y, s, dt, mask, nb, nt, plan)
    q1 = _tri(r11, bh, adjoint=True)
    q2 = _tri(r22, th - jnp.matmul(jnp.conj(r12).T, q1, precision=HIGHEST), adjoint=True)
    return q1, q2


@partial(jax.custom_jvp, nondiff_argnums=(7,))
def _refit(y, s, s0, dt, mask, blocks, c0, plan):
    """Refit at s == s0 in value, without differentiating QR or basis exponentials."""
    return c0


@_refit.defjvp
def _refit_jvp(plan, primals, tangents):
    y, s, s0, dt, mask, blocks, c0 = primals
    dy, ds = tangents[:2]
    r11, r12, _r13, r22, r23, nb, nt = blocks
    q1, _ = _project(dy, s0, dt, mask, blocks, plan)
    rhs = q1 - jnp.matmul(r12, (nt * ds)[:, None] * c0, precision=HIGHEST)
    residual = jnp.matmul(jnp.conj(r22).T, r23, precision=HIGHEST)
    correction = (jnp.conj(ds) * nt / nb)[:, None] * residual
    dc = (_tri(r11, rhs) + _tri(r11, _tri(r11, correction, adjoint=True))) / nb[:, None]
    return _refit(*primals, plan), dc * mask[:, None]


def fit_window(y, s0, dt, mask, *, pole_step, constant_poles=False):
    """Return poles, residues, initial residues and the zero pole step for one window.

    The private byte target is read while tracing; existing jitted functions
    retain their traced balanced plan. Every real block has at least 2K + C
    rows when the window permits, sizes differ by at most one, and a shorter
    window is one block. QR and products share that plan. No M by K basis survives a scan iteration.
    ``pole_step(q2, ghat, weights, mask)`` is the caller's frozen pole step,
    applied to the compressed window ``Q2^H Y`` and the frozen ``Ghat``.
    """
    frozen_y, frozen_s = jax.lax.stop_gradient(y), jax.lax.stop_gradient(s0)
    plan = _block_plan(y.shape[0], 2 * s0.size + y.shape[1], s0.dtype)
    blocks = jax.tree_util.tree_map(jax.lax.stop_gradient, _factor(frozen_y, frozen_s, dt, mask, plan))
    r11, _r12, r13, r22, _r23, nb, nt = blocks
    c0 = jax.lax.stop_gradient(_tri(r11, r13) / nb[:, None] * mask[:, None])
    weights = jax.lax.stop_gradient(1 / jnp.sqrt(jnp.mean(jnp.abs(y)**2, axis=0))).astype(s0.dtype)
    ghat = (r22[:, None, :] * (c0.T * weights[:, None] * nt[None, :])[None, :, :]).reshape(-1, s0.size)
    ghat = jax.lax.stop_gradient(ghat)
    if constant_poles:
        ds = jnp.zeros_like(s0)
    else:
        _, q2 = _project(y, frozen_s, dt, mask, blocks, plan)
        ds = pole_step(q2, ghat, weights, mask)
    s = s0 + ds
    c = _refit(y, s, frozen_s, dt, mask, blocks, c0, plan)
    return s, c, c0, ds
