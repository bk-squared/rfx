"""The ring-down completion as a JAX map whose derivative is right.

Physics. After the pulse, every port voltage and current is a finite sum of the
same damped oscillations, ``y_n = sum_k r_k lambda_k**n``; the completion
(``rfx.ringdown``) identifies them on the last part of the record and adds the
unrecorded part of each spectrum in closed form. A gradient of an objective
built on that completed spectrum must be the derivative of the INFINITE-record
spectrum: when a design parameter moves a resonance, the frequency-shift term
``r n dt s' lambda**n`` is what a cut record lacks and the completion supplies.

The map, ``y (n, C) -> completed spectra (nf, C)``, is the research lane's E3':

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
import warnings

import jax
import numpy as np

INERT_LAMBDA = 0.5
DFT_BLOCK = 1024
MUTATIONS = (None, "poles_constant", "hold_residues", "device_tail_arg")
STATUS_OK, STATUS_FAILED, STATUS_OVER_BUDGET = 0, 1, 2
STATUS_INCONSISTENT, STATUS_PRECONDITION = 3, 4


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



def host_poles(identify_window, window, k_max, dt, freqs):
    """``(s, mask, status, tail_arg, count)`` from the host, traced constants.

    ``identify_window(w)`` gets the window as a numpy array of the traced
    window's own dtype and shape, bit for bit, and returns the kept poles
    (complex, per second, at the original step), or (poles, completion mask).
    Mask value 2 retains an unresolved pole in the fit but omits its tail. ``window`` is
    ``stop_gradient``-ed here. Returns ``s (k_max,)`` in the working complex
    dtype (the real poles first, then inert ones, ``lambda = 0.5`` per step),
    ``mask (k_max,)`` in int32, ``status`` (int32):
    :data:`STATUS_OK`, :data:`STATUS_FAILED` (the identification raised) or
    :data:`STATUS_OVER_BUDGET` (more than ``k_max`` poles), on either failure
    with an all-zero mask; and ``tail_arg (nf, k_max)``, ``s dt + log z`` at
    ``freqs`` formed in float64 on the host and then cast (see the module
    docstring, TAIL). ``count`` is the kept count before padding or budget
    rejection, or -1 if identification raised; it survives an empty failure mask.
    """
    import jax
    import jax.numpy as jnp

    from rfx.ringdown import IdentificationMemoryRefused

    rdt, cdt = working_dtypes()
    K = int(k_max)
    part = np.dtype(rdt)
    n_words = part.itemsize // 4
    in_dtype = np.dtype(window.dtype)
    shape = tuple(int(d) for d in window.shape)
    dt = float(dt)
    inert = complex(np.log(INERT_LAMBDA)) / dt
    logz = -2j * np.pi * np.asarray(freqs, dtype=np.float64) * dt
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
            identified = identify_window(w)
            poles, complete = identified if isinstance(identified, tuple) else (identified, True)
            s = np.asarray(poles, dtype=np.complex128).ravel()
        except IdentificationMemoryRefused:
            raise
        except (TimeoutError,): raise
        except Exception:  # noqa: BLE001
            s, status = np.zeros(0, dtype=np.complex128), STATUS_FAILED
        count = -1 if status == STATUS_FAILED else s.size
        if s.size > K:
            s, status = np.zeros(0, dtype=np.complex128), STATUS_OVER_BUDGET
        s_out[:s.size] = s
        if s.size:
            mask[:s.size] = np.where(complete, 1, 2)
        arg = s_out[None, :] * dt + logz[:, None]
        return (words(s_out), mask, np.asarray(status, dtype=np.int32), words(arg),
                np.asarray(count, dtype=np.int32))

    w_in = jax.lax.bitcast_convert_type(jax.lax.stop_gradient(window), jnp.uint32)
    shapes = (jax.ShapeDtypeStruct((K, 2, n_words), jnp.uint32),
              jax.ShapeDtypeStruct((K,), jnp.int32),
              jax.ShapeDtypeStruct((), jnp.int32),
              jax.ShapeDtypeStruct((nf, K, 2, n_words), jnp.uint32),
              jax.ShapeDtypeStruct((), jnp.int32))
    s_words, mask_i, status, arg_words, count = jax.pure_callback(
        host, shapes, w_in, **_callback_keywords())

    def from_words(x):
        parts = jax.lax.bitcast_convert_type(x[..., 0] if n_words == 1 else x, rdt)
        return jax.lax.complex(parts[..., 0], parts[..., 1]).astype(cdt)

    return from_words(s_words), mask_i, status, from_words(arg_words), count





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
    with jax.default_matmul_precision("highest"):
        Q, R = jnp.linalg.qr(A_aug, mode="reduced")
    return Q, R, norms


def _solve_qr(Q, R, norms, mask, b):
    """Least-squares solution for the factorised system against ``[b; 0]``, masked."""
    import jax.numpy as jnp
    from rfx._internal._precision import HIGHEST
    from rfx.core.jax_utils import complex_solve_triangular
    K = R.shape[0]
    b = b.astype(Q.dtype)
    b_aug = jnp.concatenate([b, jnp.zeros((K, b.shape[1]), dtype=b.dtype)], axis=0)
    x = complex_solve_triangular(R, jnp.matmul(jnp.conj(Q).T, b_aug, precision=HIGHEST),
                         lower=False)
    return (x / norms.astype(x.dtype)[:, None]) * jnp.real(mask).astype(x.dtype)[:, None]


def lstsq_masked(A, b, mask):
    Q, R, norms = _qr_masked(A, mask)
    return _solve_qr(Q, R, norms, mask, b)


@jax.custom_jvp
def _first_order_only(Yw, w):
    """``w``, unchanged; a second derivative through it is NaN, with a warning.

    The rule of :func:`_pole_step` passes the channel weights through here.
    Its JVP rule runs only when the record ``Yw`` carries a derivative inside
    that rule, that is, when a second differentiation reaches the
    identification window. It then emits
    :class:`rfx.errors.RingdownSecondDerivativeWarning` while tracing and
    returns a NaN tangent linear in the record's tangent, so the NaN survives
    transposition and reaches only what reads the pole step's derivative.
    """
    return w


@_first_order_only.defjvp
def _first_order_only_jvp(primals, tangents):
    import jax.numpy as jnp
    from rfx.errors import RingdownSecondDerivativeWarning
    _Yw, w = primals
    Yw_dot = tangents[0]
    warnings.warn(
        "a second derivative reached the ring-down completion of forward(ringdown=...): "
        "the completion moves its poles by a first-order step, so a second derivative of a "
        "completed output (ringdown.s_params, s_params_long, poles, probe_spectra) is NaN. "
        "First derivatives, and every derivative of an objective that reads no completed "
        "output, are exact.",
        RingdownSecondDerivativeWarning, stacklevel=2)
    return w, jnp.nan * jnp.sum(Yw_dot, axis=0)


@jax.custom_jvp
def _pole_step(Yw, A, w, mask_c):
    """The frozen Gauss-Newton pole step: zero in value, ``Dmat dYw`` in derivative.

    ``Yw (r, C)`` is the compressed window ``Q2^H Y``, ``A (r*C, K)``
    the frozen compressed weighted Jacobian ``Ghat`` and ``w (C,)`` the
    frozen channel weights. The projection is linear in the window with
    its poles, factors and column norms frozen. The value is the
    least-squares solution against ``(Yw - Yw) w``, which is zero unless the
    data are not finite. The first derivative is the least-squares solution
    against ``dYw w``; a second derivative is NaN (:func:`_first_order_only`).
    The rule computes its value by calling this function again, so the value
    keeps its first derivative under a second differentiation.
    """
    M, C = int(Yw.shape[0]), int(Yw.shape[1])
    return lstsq_masked(A, ((Yw - Yw) * w[None, :]).reshape(M * C, 1), mask_c)[:, 0]


@_pole_step.defjvp
def _pole_step_jvp(primals, tangents):
    Yw, A, w, mask_c = primals
    Yw_dot = tangents[0]
    M, C = int(Yw.shape[0]), int(Yw.shape[1])
    g = _first_order_only(Yw, w)
    ds = _pole_step(Yw, A, w, mask_c)
    ds_dot = lstsq_masked(A, (Yw_dot * g[None, :]).reshape(M * C, 1), mask_c)[:, 0]
    return ds, ds_dot



def completion(y, dt, freqs, n_start, s0, mask, tail_arg, *, plain=None,
               mutation=None, return_aux=False):
    """Completed spectra of the record ``y (n, C)`` (samples 0..n-1).

    ``s0, mask, tail_arg`` come from :func:`host_poles` for the window
    ``[n_start, n)`` and the bins ``freqs``; the tail starts after sample
    ``n - 1``. ``plain`` may carry :func:`plain_dft` of ``y`` at ``freqs`` when
    the caller already has it. See the module docstring for the value and the
    derivative. ``mutation`` exists only for the unit tests' revived-defect
    checks (:data:`MUTATIONS`).

    Derivatives. First derivatives are exact, and so is every derivative of
    an objective that reads no completed output. The poles move by a
    first-order step (:func:`_pole_step`), so an entry of a second-order form
    (``jax.hessian``, or ``jax.grad``, ``jax.jvp``, ``jax.jacfwd``,
    ``jax.jacrev`` nested) that needs the completed spectrum's second
    derivative is NaN. An entry of a joint second-order form that needs only
    its first derivative is exact or NaN, never another finite number.
    The warning :class:`rfx.errors.RingdownSecondDerivativeWarning` shows
    whenever a second differentiation reaches the identification window,
    also when the objective reads no completed output.
    """
    from rfx.measurement.ringdown_tail_jax import plain_dft, tail_dft
    import jax
    import jax.numpy as jnp

    rdt, cdt = working_dtypes()
    if mutation not in MUTATIONS:
        raise ValueError(f"mutation {mutation!r} not in {MUTATIONS}")
    n = int(y.shape[0])
    n_start = int(n_start)
    if not (0 <= n_start < n):
        raise ValueError(f"window [{n_start}, {n}) outside {n} samples")
    K = int(s0.shape[0])
    y = y.astype(rdt)
    yw = y[n_start:]
    Yw = yw.astype(cdt)
    tail_mask = mask == 1
    mask = (mask > 0).astype(rdt)
    mask_c = mask.astype(cdt)

    from rfx.measurement.ringdown_blocked import fit_window

    s, c, c0, ds = fit_window(Yw, s0, dt, mask_c, pole_step=_pole_step,
                              constant_poles=mutation == "poles_constant")
    if mutation == "hold_residues":
        c = jax.lax.stop_gradient(c)

    if plain is None:
        plain = plain_dft(y, dt, freqs)
    spectra = plain + tail_dft(s, c, n_start, n - 1, dt, freqs,
                               None if mutation == "device_tail_arg" else tail_arg, mask=tail_mask)
    if not return_aux:
        return spectra
    aux = {"s_pencil": s0, "mask": mask, "delta_s": ds, "s": s, "c": c, "c0": c0,
           "n_real": jnp.sum(mask), "k_max": K, "n_start": n_start, "n_record": n}
    return spectra, aux
