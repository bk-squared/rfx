"""The ring-down completion as a JAX map whose derivative is right (issue #1254, PR 3).

Physics. After the pulse, every port voltage and current is a finite sum of the
same damped oscillations, ``y_n = sum_k r_k lambda_k**n``; the completion
(``rfx.ringdown``) identifies them on the last part of the record and adds the
unrecorded part of each spectrum in closed form. A gradient of an objective
built on that completed spectrum must be the derivative of the INFINITE-record
spectrum: when a design parameter moves a resonance, the frequency-shift term
``r n dt s' lambda**n`` is what a cut record lacks and the completion supplies.

The map, ``y (n, C) -> completed spectra (nf, C)``, is the research lane's E3'
(rfx-archive ``20260924_acc_E3_reverse_mode_completion_results.md``):

* VALUE: the pencil poles ``s0`` from the host (``jax.pure_callback`` on
  ``stop_gradient`` data, identified by :func:`rfx.ringdown.identify`, the
  same rules ``run(ringdown=...)`` uses) and their least-squares residues at
  the original rate over the window; the closed-form tail; the plain DFT of the
  record. No pole step is taken, so the value is ``run()``'s completion.
* DERIVATIVE of the poles: the Gauss-Newton implicit derivative at ``s0`` with
  the Jacobian and the channel weights frozen at their primal values,
  ``s(y) = s0 + Dmat (y - stop_gradient(y))``, ``Dmat dy`` the least-squares
  solution of ``A ds = W^(1/2) dy`` with ``A = W^(1/2) P_B^perp D`` (variable
  projection, Kaufman's Jacobian, ``D[:, ch, k] = m dt exp(s_k m dt) c_k,ch``,
  ``W`` the 1/RMS channel weights) held constant. The right-hand side is zero
  in value, so ``s(y) == s0`` bit for bit.
* RESIDUES re-fitted differentiably at ``s(y)`` (QR of the traced basis).
* TAIL. ``1 - lambda z = -expm1(s dt + log z)``: near a resonance the two
  imaginary parts, each about ``omega dt``, cancel to about ``alpha dt``, so in
  complex64 the difference loses ``omega / alpha = 2 Q`` of its digits (a Q
  1e4 mode completed 1.9e-3 off, its derivative 3.8e-3). The host forms
  ``s0 dt + log z`` in float64 with the poles; the device adds only the
  zero-valued ``ds dt``, so the value keeps float32's relative precision at
  any Q and the derivative is unchanged.

Static shapes. ``jax.jit`` needs a fixed pole count: the host returns ``k_max``
poles, the real ones first, padded with an inert pole (``lambda = 0.5`` per
step) and a mask. Every least-squares problem is solved on
``[A * mask; diag(1 - mask)]`` against ``[b; 0]``: a padded column is a unit
vector in its own extra row, so the real columns see exactly their unpadded
problem, and the solution is multiplied by the mask, so a padded residue and a
padded pole derivative are exactly 0.0 in value and in derivative.

Transport. The window goes to the host as ``uint32`` words of its samples and
the poles come back as ``uint32`` words of their real and imaginary parts
(``lax.bitcast_convert_type``): JAX 0.4.33 canonicalises a callback's arguments
on the host-callback thread, where a scoped x64 context is not active, which
rounded float64 data to float32 and rejected a complex128 return (the lane's
run 369367264106). ``uint32`` is what it is under any x64 setting, and both
views are exact.

Precision. The map computes in float64/complex128 when the CALLER has enabled
JAX x64, otherwise in float32/complex64. This module never enables x64.
Constants (the DFT kernels, ``z``, ``z**(N+1)``) are built on the host in
float64 and cast.
"""

from __future__ import annotations

import inspect

import numpy as np

#: The inert padding pole, per step (``s = log(0.5) / dt``). Its basis column
#: underflows toward zero within a few hundred samples and is masked anyway.
INERT_LAMBDA = 0.5
#: Block length of the plain DFT (host-built float64 kernel, cast per block).
DFT_BLOCK = 1024
#: Mutations, for the unit tests' revived-defect checks only:
#: ``"poles_constant"`` drops the pole derivative (the frequency-shift term is
#: lost); ``"hold_residues"`` stops the gradient of the residues;
#: ``"device_tail_arg"`` forms ``s dt + log z`` on the device in the working
#: precision (the complex64 cancellation at high Q).
MUTATIONS = (None, "poles_constant", "hold_residues", "device_tail_arg")
#: Host status codes carried back with the poles.
STATUS_OK, STATUS_FAILED, STATUS_OVER_BUDGET = 0, 1, 2
#: Set by the caller (not the host) when its in-program consistency check fails.
STATUS_INCONSISTENT = 3


def working_dtypes():
    """``(real, complex)`` dtypes of the map: float64 only if the caller enabled x64."""
    import jax
    import jax.numpy as jnp
    if bool(jax.config.read("jax_enable_x64")):
        return jnp.float64, jnp.complex128
    return jnp.float32, jnp.complex64


def _callback_keywords():
    """JAX >= 0.4.34 takes ``vmap_method``; 0.4.33 knows only ``vectorized``
    and would pass an unknown keyword to the host function as an argument."""
    import jax
    if "vmap_method" in inspect.signature(jax.pure_callback).parameters:
        return {"vmap_method": "sequential"}
    return {"vectorized": False}


# ---------------------------------------------------------------------------
# Poles from the host
# ---------------------------------------------------------------------------

def host_poles(identify_window, window, k_max, dt, freqs):
    """``(s, mask, status, tail_arg)`` from the host, constants of the traced program.

    ``identify_window(w)`` gets the window as a numpy array of the traced
    window's own dtype and shape, bit for bit, and returns the kept poles
    (complex, per second, at the original step). ``window`` is
    ``stop_gradient``-ed here. Returns ``s (k_max,)`` in the working complex
    dtype (the real poles first, then inert ones, ``lambda = 0.5`` per step),
    ``mask (k_max,)`` in the working real dtype, ``status`` (int32):
    :data:`STATUS_OK`, :data:`STATUS_FAILED` (the identification raised) or
    :data:`STATUS_OVER_BUDGET` (more than ``k_max`` poles), on either failure
    with an all-zero mask; and ``tail_arg (nf, k_max)``, ``s dt + log z`` at
    ``freqs`` formed in float64 on the host and then cast (see the module
    docstring, TAIL).
    """
    import jax
    import jax.numpy as jnp

    rdt, cdt = working_dtypes()
    K = int(k_max)
    part = np.dtype(rdt)
    n_words = part.itemsize // 4
    in_dtype = np.dtype(window.dtype)
    shape = tuple(int(d) for d in window.shape)
    dt = float(dt)
    inert = complex(np.log(INERT_LAMBDA)) / dt
    logz = -2j * np.pi * np.asarray(freqs, dtype=np.float64) * dt      # (nf,)
    nf = int(logz.size)

    def words(z):
        """Complex float64 values as uint32 words of their working-precision parts."""
        parts = np.ascontiguousarray(np.stack([z.real, z.imag], axis=-1).astype(part))
        return parts.view(np.uint32).reshape(z.shape + (2, n_words))

    def host(words_in):
        w = np.ascontiguousarray(np.asarray(words_in, dtype=np.uint32)).view(in_dtype)
        w = w.reshape(shape)
        s_out = np.full(K, inert, dtype=np.complex128)
        mask = np.zeros(K, dtype=np.int32)
        status = STATUS_OK
        try:
            s = np.asarray(identify_window(w), dtype=np.complex128).ravel()
        except Exception:  # noqa: BLE001  (reported by the host report, not here)
            s, status = np.zeros(0, dtype=np.complex128), STATUS_FAILED
        if s.size > K:
            s, status = np.zeros(0, dtype=np.complex128), STATUS_OVER_BUDGET
        s_out[:s.size] = s
        mask[:s.size] = 1
        arg = s_out[None, :] * dt + logz[:, None]                          # float64
        return (words(s_out), mask, np.asarray(status, dtype=np.int32), words(arg))

    w_in = jax.lax.bitcast_convert_type(jax.lax.stop_gradient(window), jnp.uint32)
    shapes = (jax.ShapeDtypeStruct((K, 2, n_words), jnp.uint32),
              jax.ShapeDtypeStruct((K,), jnp.int32),
              jax.ShapeDtypeStruct((), jnp.int32),
              jax.ShapeDtypeStruct((nf, K, 2, n_words), jnp.uint32))
    s_words, mask_i, status, arg_words = jax.pure_callback(host, shapes, w_in,
                                                           **_callback_keywords())

    def from_words(x):
        parts = jax.lax.bitcast_convert_type(x[..., 0] if n_words == 1 else x, rdt)
        return jax.lax.complex(parts[..., 0], parts[..., 1]).astype(cdt)

    # The pole is the unit that crossed from the host: a complex s (1/s) in
    # the working precision.
    return from_words(s_words), mask_i.astype(rdt), status, from_words(arg_words)


# ---------------------------------------------------------------------------
# DFTs
# ---------------------------------------------------------------------------

def plain_dft(y, dt, freqs, block=DFT_BLOCK):
    """``dt * sum_n y_n exp(-j 2 pi f n dt)``, ``y (n, C)`` -> ``(nf, C)``.

    :func:`rfx.ringdown.plain_dft`'s convention. Blocked so no ``nf x n``
    kernel is formed: ``exp(-j w (b B + m)) = exp(-j w b B) exp(-j w m)``,
    both factors built on the host in float64 and cast to the working dtype.
    """
    import jax.numpy as jnp
    rdt, cdt = working_dtypes()
    n, C = int(y.shape[0]), int(y.shape[1])
    freqs = np.asarray(freqs, dtype=np.float64)
    B = int(min(block, n))
    nb = -(-n // B)
    w = -2.0 * np.pi * freqs * float(dt)
    k_blk = np.exp(1j * np.outer(w, np.arange(B, dtype=np.float64)))           # (nf, B)
    z_off = np.exp(1j * np.outer(np.arange(nb, dtype=np.float64) * B, w))      # (nb, nf)
    yp = jnp.zeros((nb * B, C), dtype=rdt).at[:n].set(y.astype(rdt))
    yb = yp.reshape(nb, B, C).astype(cdt)
    inner = jnp.einsum("fm,bmc->bfc", jnp.asarray(k_blk, dtype=cdt), yb)
    return float(dt) * jnp.einsum("bf,bfc->fc", jnp.asarray(z_off, dtype=cdt), inner)


def tail_dft(s, c, n_ref, n_last, dt, freqs, tail_arg=None):
    """:func:`rfx.ringdown.tail_dft` in JAX.

    ``dt sum_k c_k lambda_k**(N+1-n_ref) z**(N+1) / (1 - lambda_k z)`` with
    ``1 - lambda z = -expm1(s dt - j 2 pi f dt)``. ``c (K, C)``; returns ``(nf, C)``.
    ``tail_arg (nf, K)`` is ``s0 dt + log z`` from :func:`host_poles`, formed in
    float64; the device adds ``(s - stop_gradient(s)) dt`` to it, zero in value.
    ``None`` forms the sum on the device in the working precision.
    """
    import jax
    import jax.numpy as jnp
    _rdt, cdt = working_dtypes()
    freqs = np.asarray(freqs, dtype=np.float64)
    dt = float(dt)
    zN1 = jnp.asarray(np.exp(-2j * np.pi * freqs * dt * (int(n_last) + 1)), dtype=cdt)
    sdt = s * dt
    if tail_arg is None:
        logz = jnp.asarray(-2j * np.pi * freqs * dt, dtype=cdt)               # (nf,)
        one_minus_lz = -jnp.expm1(sdt[None, :] + logz[:, None])               # (nf, K)
    else:
        dsdt = (s - jax.lax.stop_gradient(s)) * dt
        one_minus_lz = -jnp.expm1(tail_arg + dsdt[None, :])
    amp = jnp.exp(sdt * float(int(n_last) + 1 - int(n_ref)))[:, None] * c     # (K, C)
    return dt * ((zN1[:, None] / one_minus_lz) @ amp)


# ---------------------------------------------------------------------------
# Masked least squares (QR on normalised columns, padded columns decoupled)
# ---------------------------------------------------------------------------

def _basis(s, M, dt):
    """``B[m, k] = exp(s_k m dt)``, ``m = 0..M-1`` (the window's first sample is m = 0)."""
    import jax.numpy as jnp
    rdt, _cdt = working_dtypes()
    t = jnp.arange(int(M), dtype=rdt) * float(dt)
    return jnp.exp(t[:, None] * s[None, :]), t


def _qr_masked(A, mask):
    """QR of ``[A_n * mask; diag(1 - mask)]``, ``A_n`` = ``A`` with unit-norm columns.

    Returns ``(Q, R, norms)``; ``norms`` are ``stop_gradient``-ed (a column
    scaling does not change a full-rank least-squares solution, so treating it
    as a constant leaves the derivative unchanged) and 1 on padded columns.
    """
    import jax
    import jax.numpy as jnp
    mask = jnp.real(mask)
    A = A * mask.astype(A.dtype)[None, :]
    norms = jnp.sqrt(jnp.sum(jnp.abs(A) ** 2, axis=0))
    norms = jax.lax.stop_gradient(jnp.where(mask > 0, norms, 1.0))
    aug = jnp.diag((1.0 - mask).astype(A.dtype))
    A_aug = jnp.concatenate([A / norms.astype(A.dtype)[None, :], aug], axis=0)
    Q, R = jnp.linalg.qr(A_aug, mode="reduced")
    return Q, R, norms


def _solve_qr(Q, R, norms, mask, b):
    """Least-squares solution for the factorised system against ``[b; 0]``, masked."""
    import jax.numpy as jnp
    from jax.scipy.linalg import solve_triangular
    K = R.shape[0]
    b = b.astype(Q.dtype)
    b_aug = jnp.concatenate([b, jnp.zeros((K, b.shape[1]), dtype=b.dtype)], axis=0)
    x = solve_triangular(R, jnp.conj(Q).T @ b_aug, lower=False)
    return (x / norms.astype(x.dtype)[:, None]) * jnp.real(mask).astype(x.dtype)[:, None]


def lstsq_masked(A, b, mask):
    Q, R, norms = _qr_masked(A, mask)
    return _solve_qr(Q, R, norms, mask, b)


# ---------------------------------------------------------------------------
# The completion map
# ---------------------------------------------------------------------------

def completion(y, dt, freqs, n_start, s0, mask, tail_arg, *, plain=None,
               mutation=None, return_aux=False):
    """Completed spectra of the record ``y (n, C)`` (samples 0..n-1).

    ``s0, mask, tail_arg`` come from :func:`host_poles` for the window
    ``[n_start, n)`` and the bins ``freqs``; the tail starts after sample
    ``n - 1``. ``plain`` may carry :func:`plain_dft` of ``y`` at ``freqs`` when
    the caller already has it. See the module docstring for the value and the
    derivative. ``mutation`` exists only for the unit tests' revived-defect
    checks (:data:`MUTATIONS`).
    """
    import jax
    import jax.numpy as jnp

    rdt, cdt = working_dtypes()
    if mutation not in MUTATIONS:
        raise ValueError(f"mutation {mutation!r} not in {MUTATIONS}")
    n, C = int(y.shape[0]), int(y.shape[1])
    n_start = int(n_start)
    if not (0 <= n_start < n):
        raise ValueError(f"window [{n_start}, {n}) outside {n} samples")
    M = n - n_start
    K = int(s0.shape[0])
    y = y.astype(rdt)
    yw = y[n_start:]
    Yw = yw.astype(cdt)
    mask_c = mask.astype(cdt)

    # the value: least-squares residues at the host poles
    B0, t = _basis(s0, M, dt)
    B0m = B0 * mask_c[None, :]
    Q0, R0, nrm0 = _qr_masked(B0, mask_c)
    c0 = _solve_qr(Q0, R0, nrm0, mask_c, Yw)                                   # (K, C)

    if mutation == "poles_constant":
        ds = jnp.zeros_like(s0)
    else:
        # the frozen Gauss-Newton implicit derivative of the poles
        rms = jnp.sqrt(jnp.mean(jnp.abs(yw) ** 2, axis=0))
        w = jax.lax.stop_gradient(1.0 / rms).astype(cdt)                       # (C,)
        D = t.astype(cdt)[:, None, None] * B0m[:, None, :] * (c0.T * w[:, None])[None, :, :]
        QB = Q0[:M] * mask_c[None, :]
        PD = D - jnp.einsum("mk,kcj->mcj", QB, jnp.einsum("mk,mcj->kcj", jnp.conj(QB), D))
        A = jax.lax.stop_gradient(PD.reshape(M * C, K))
        dyw = ((Yw - jax.lax.stop_gradient(Yw)) * w[None, :]).reshape(M * C, 1)
        ds = lstsq_masked(A, dyw, mask_c)[:, 0]
    s = s0 + ds

    # residues re-fitted differentiably at s(y); in value, the fit at s0
    B, _t = _basis(s, M, dt)
    c = lstsq_masked(B, Yw, mask_c)
    if mutation == "hold_residues":
        c = jax.lax.stop_gradient(c)

    if plain is None:
        plain = plain_dft(y, dt, freqs)
    spectra = plain + tail_dft(s, c, n_start, n - 1, dt, freqs,
                               None if mutation == "device_tail_arg" else tail_arg)
    if not return_aux:
        return spectra
    aux = {"s_pencil": s0, "mask": mask, "delta_s": ds, "s": s, "c": c, "c0": c0,
           "n_real": jnp.sum(mask), "k_max": K, "n_start": n_start, "n_record": n}
    return spectra, aux
