"""Simulation-time estimates for resonance sensitivities (arXiv:2610.28496)."""

import math

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import brentq


def sensitivity_truncation_error(t_over_tau, detuning_halfwidths=0.0):
    """Estimate relative sensitivity error from a finite simulation time, arXiv:2610.28496.

    For a parameter that moves a resonance, the sensitivity's late-time part
    decays as ``t exp(-t/tau)``. At resonance its truncation error is
    ``(1 + T/tau) exp(-T/tau)``, against ``exp(-T/tau)`` for the resonant
    part of S. Here ``tau = 2Q/omega`` is the decay time of the slowest mode
    the quantity depends on, and ``d = (f - f_r) / HWHM`` is detuning in
    half-widths. The estimate is ``exp(-x) hypot(1 + x, d*x)``, ``x = T/tau``.

    Scope: a single dominant mode, residue term dropped, pole shift dominating.
    At lower Q the true error lies above the estimate; at transmission zeros
    it can exceed it. Use :func:`rfx.ringdown.gradient_witness` to test an
    actual run.

    Inputs broadcast using NumPy, or JAX if either input is a JAX array.
    Floating dtypes follow the inputs, including float32; Python float scalars
    return a Python float. Concrete inputs raise ValueError for nonpositive or
    non-finite ``t_over_tau``, or non-finite detuning. Traced JAX inputs cannot
    raise: invalid elements return NaN instead (their gradient is 0, not NaN).
    Inputs must be real.
    """
    inputs = (t_over_tau, detuning_halfwidths)
    traced = any(isinstance(v, jax.core.Tracer) for v in inputs)
    xp = jnp if any(isinstance(v, (jax.Array, jax.core.Tracer)) for v in inputs) else np
    dtype = xp.result_type(t_over_tau, detuning_halfwidths, 0.0)
    x, d = (xp.asarray(v, dtype=dtype) for v in inputs)
    if xp.iscomplexobj(x) or xp.iscomplexobj(d):
        raise ValueError("t_over_tau and detuning_halfwidths must be real")
    valid = (x > 0) & xp.isfinite(x) & xp.isfinite(d)
    if not traced and not np.all(np.asarray(valid)):
        raise ValueError("t_over_tau must be positive and finite; detuning_halfwidths must be finite")
    result = xp.exp(-x) * xp.hypot(1 + x, d * x)
    if traced:
        result = xp.where(valid, result, xp.nan)
    if xp is np and all(isinstance(v, (int, float)) for v in inputs):
        return float(result)
    return result


def required_t_over_tau(error=0.01, detuning_halfwidths=0.0):
    """Return the host scalar T/tau needed for a sensitivity error below ``error``.

    The estimate and its single-mode scope are described in
    :func:`sensitivity_truncation_error` (arXiv:2610.28496). For example,
    ``required_t_over_tau(0.01)`` is about 6.64, against ``ln(100) = 4.6``
    for the resonant part of S. Test an actual run with
    :func:`rfx.ringdown.gradient_witness`.

    For 0 < error < 1 there is one root: any rising branch has error >= 1.
    The falling branch is strictly decreasing, in particular for x >= 1.
    Brent's bracket starts at lo = 1e-12, hi = 1; hi doubles until the
    estimate is below error. The root has relative accuracy <= 1e-12.
    Non-finite detuning and error outside (0, 1) raise ValueError.
    Float64 limits: error below about 1e-300 or |detuning| above about 1e300
    under- or overflows and the root is not meaningful.
    """
    error, d = float(error), float(detuning_halfwidths)
    if not 0 < error < 1 or not math.isfinite(d):
        raise ValueError("error must be in (0, 1); detuning_halfwidths must be finite")

    def residual(x):
        return sensitivity_truncation_error(x, d) - error

    lo, hi = 1e-12, 1.0
    while residual(hi) >= 0:
        hi *= 2
    return float(brentq(residual, lo, hi, xtol=np.finfo(float).tiny, rtol=1e-13))
