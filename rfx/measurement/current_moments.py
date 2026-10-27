"""Far-field transforms and record conventions for current moments."""

from __future__ import annotations

from rfx._precision import HIGHEST
from rfx.core.yee import EPS_0
from rfx.current_moments import CurrentMomentMonitor
from rfx.current_moments import ETA_0
from rfx.current_moments import UNIFORM_PLANE_STAMP_STEPS
from rfx.current_moments import _reduce_to_blocks
from rfx.current_moments import slab_e_snapshot
import jax.numpy as jnp
import numpy as np


def block_far_field_jax(theta, phi, centres, P, Q, T, k, order, eta=ETA_0):
    """The far field of a set of blocks, in JAX, for the differentiable arm.

    Each block contributes ``exp(+jk rhat.c) [P + jk (rhat.Q)
    - (k^2/2)(rhat rhat : T)]`` to the electric radiation vector — the Taylor
    expansion of ``exp(+jk rhat.r)`` about the block's own centre. The same
    expression as :func:`block_far_field_np`; the tests pin the two together
    rather than trusting that two spellings agree.
    """
    r_hat, th_hat, ph_hat, shape = _direction_frame_np(theta, phi)
    r_hat = jnp.asarray(r_hat)
    th_hat = jnp.asarray(th_hat)
    ph_hat = jnp.asarray(ph_hat)
    centres = jnp.asarray(centres)
    ph = jnp.exp(1j * k * jnp.matmul(r_hat, centres.T, precision=HIGHEST))  # (n_dir, n_blocks)
    N = jnp.matmul(ph, jnp.asarray(P), precision=HIGHEST)
    if order >= 1 and Q is not None:
        Qj = jnp.asarray(Q)
        for a in range(3):
            N = N + 1j * k * r_hat[:, a, None] * jnp.matmul(ph, Qj[:, a, :], precision=HIGHEST)
    if order >= 2 and T is not None:
        Tj = jnp.asarray(T)
        for a in range(3):
            for b in range(3):
                N = N + (-(k ** 2) / 2.0) * (
                    r_hat[:, a, None] * r_hat[:, b, None]
                    * jnp.matmul(ph, Tj[:, a, b, :], precision=HIGHEST))
    jk4pi = 1j * k / (4.0 * np.pi)
    e_th = -jk4pi * eta * jnp.sum(N * th_hat, axis=1)
    e_ph = -jk4pi * eta * jnp.sum(N * ph_hat, axis=1)
    return e_th.reshape(shape), e_ph.reshape(shape)


def block_far_field_np(theta, phi, centres, P, Q, T, k, order, eta=ETA_0):
    """Host twin of :func:`block_far_field_jax`, same expression."""
    r_hat, th_hat, ph_hat, shape = _direction_frame_np(theta, phi)
    centres = np.asarray(centres, dtype=np.float64)
    ph = np.exp(1j * k * (r_hat @ centres.T))
    N = ph @ np.asarray(P, dtype=np.complex128)
    if order >= 1 and Q is not None:
        Q = np.asarray(Q, dtype=np.complex128)
        for a in range(3):
            N += 1j * k * r_hat[:, a, None] * (ph @ Q[:, a, :])
    if order >= 2 and T is not None:
        T = np.asarray(T, dtype=np.complex128)
        for a in range(3):
            for b in range(3):
                N += (-(k ** 2) / 2.0) * (r_hat[:, a, None]
                                          * r_hat[:, b, None] * (ph @ T[:, a, b, :]))
    jk4pi = 1j * k / (4.0 * np.pi)
    e_th = -jk4pi * eta * np.sum(N * th_hat, axis=1)
    e_ph = -jk4pi * eta * np.sum(N * ph_hat, axis=1)
    return e_th.reshape(shape), e_ph.reshape(shape)


def end_of_record_moments(monitor: CurrentMomentMonitor, state, n_steps, dt):
    """The one term a finite record leaves behind, as block moments.

    The post-processing route's frequency-domain current drops
    ``eps0 E^N exp(-j w (N + 1/2) dt)`` — the electric field still standing
    when the record stopped. The time-domain accumulator keeps it, so this is
    what has to be added to the post-processed moments (not subtracted from
    these) before the two are compared.
    """
    m = monitor
    ex, ey, ez = slab_e_snapshot(state, m)
    mom = np.asarray(_reduce_to_blocks(
        m, EPS_0 * ex, EPS_0 * ey, EPS_0 * ez), dtype=np.complex128)
    freqs = np.asarray(m.freqs, dtype=np.float64)
    ph = np.exp(-2j * np.pi * freqs * (float(n_steps) + 0.5) * float(dt))
    return mom[None, :, :, :] * ph[:, None, None, None]


def to_post_processing_convention(accumulator, end_term, dt, freqs, *,
                                  plane_stamp_steps=UNIFORM_PLANE_STAMP_STEPS):
    """This module's convention mapped onto the DFT-plane route's.

    The two differ by where the current is stamped and by the end-of-record
    term, and by nothing else; both are exact algebra, not corrections. The
    current here sits at ``(n + 1/2) dt`` because it is the difference of two
    electric fields; the post-processing route inherits whatever time its
    H plane probes wrote on the state (physical default s=1/2), so::

        post_J = exp(-j w dt (s - 1/2)) * (A + eps0 E^N e^{-j w (N+1/2) dt})

    with ``s = plane_stamp_steps``. Both terms of the post-processing formula
    carry the same factor, so a run recorded with the other stamp is only
    globally phased — nothing inside it is mis-registered.
    """
    freqs = np.asarray(freqs, dtype=np.float64)
    s = float(plane_stamp_steps)
    factor = np.exp(-2j * np.pi * freqs * float(dt) * (s - 0.5))
    a = np.asarray(accumulator, dtype=np.complex128)
    e = (0.0 if end_term is None
         else np.asarray(end_term, dtype=np.complex128))
    return (a + e) * factor[:, None, None, None]


def _direction_frame_np(theta, phi):
    theta = np.atleast_1d(np.asarray(theta, dtype=np.float64))
    phi = np.atleast_1d(np.asarray(phi, dtype=np.float64))
    TH, PH = np.meshgrid(theta, phi, indexing="ij")
    st, ct = np.sin(TH), np.cos(TH)
    sp, cp = np.sin(PH), np.cos(PH)
    r_hat = np.stack([st * cp, st * sp, ct], axis=-1).reshape(-1, 3)
    th_hat = np.stack([ct * cp, ct * sp, -st], axis=-1).reshape(-1, 3)
    ph_hat = np.stack([-sp, cp, np.zeros_like(sp)], axis=-1).reshape(-1, 3)
    return r_hat, th_hat, ph_hat, TH.shape
