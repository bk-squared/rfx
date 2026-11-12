"""Shared full-sphere power integration, independent of far-field containers."""

import numpy as np

# Largest polar angle (rad) either pole may leave uncovered beyond the end
# sample's own cell. A cap of 0.02 rad is (1 - cos 0.02)/2 = 1.0e-4 of the
# sphere, 0.0009 dB for an isotropic pattern.
POLE_TOLERANCE_RAD = 0.02


def _coverage_error():
    return ValueError(
        "Total radiated power needs full-sphere coverage: theta must span "
        f"[0, pi], leaving at most {POLE_TOLERANCE_RAD} rad of polar angle "
        "uncovered beyond each end sample's own cell (the half-step cell of "
        "the gradient rule; the end sample itself for the trapezoid rule), "
        "and phi must span a full turn within one step or have exactly one "
        "sample (axisymmetric assumption). Angles must be finite and "
        "increasing; phi may repeat the 0/2*pi seam once but may not wrap "
        "past it. Realized gain with input_power needs no full-sphere coverage."
    )


def _coverage_tol(*angles):
    # Permit rounding at the coverage boundaries only.
    dtype = np.result_type(*(a.dtype for a in angles), np.float32)
    return 8 * np.finfo(dtype).eps * np.pi


def _phi_weights(phi):
    """Periodic trapezoid weights over one full turn of ``phi`` (float64).

    ``phi`` must be finite and strictly increasing, span at most a full turn
    and leave a wrap gap no larger than its larger end step. One sample is an
    axisymmetric pattern and carries the full 2*pi weight. A last sample a
    full turn after the first is the seam repeat of the first. On the unique
    samples u_0..u_{M-1}, w_i = (u_{i+1} - u_{i-1}) / 2 with u_M = u_0 + 2*pi
    and u_{-1} = u_{M-1} - 2*pi, so the weights sum to 2*pi for any spacing.
    With a seam repeat, the first sample gets half its forward interval and
    the repeat half its backward interval (the closed-interval trapezoid end
    weights); together they carry the first unique sample's weight.
    """
    phi = np.asarray(phi)
    if phi.ndim != 1 or phi.size < 1:
        raise _coverage_error()
    tol = _coverage_tol(phi)
    phi = phi.astype(np.float64)
    if not np.all(np.isfinite(phi)):
        raise _coverage_error()
    if phi.size == 1:
        return np.array([2 * np.pi])
    steps = np.diff(phi)
    span = phi[-1] - phi[0]
    if (not np.all(steps > 0) or span > 2 * np.pi + tol
            or 2 * np.pi - span > max(steps[0], steps[-1]) + tol):
        raise _coverage_error()
    seam = span >= 2 * np.pi - tol
    u = phi[:-1] if seam else phi
    u_next = np.append(u[1:], u[0] + 2 * np.pi)
    u_prev = np.insert(u[:-1], 0, u[-1] - 2 * np.pi)
    weights = (u_next - u_prev) / 2
    if seam:
        weights = np.append(weights, (phi[-1] - phi[-2]) / 2)
        weights[0] = (phi[1] - phi[0]) / 2
    return weights


def integrate_radiated_power(intensity, theta, phi, *, jax_trapezoid=False):
    """Integrate ``intensity * sin(theta)`` over solid angle per frequency.

    ``intensity`` has shape (n_freqs, n_theta, n_phi). Angles must be finite,
    increasing vectors. Each pole may leave at most ``POLE_TOLERANCE_RAD``
    of polar angle uncovered beyond the end sample's own cell. ``phi`` is
    weighted by the periodic trapezoid rule of ``_phi_weights`` on both
    paths; a single phi sample assumes an axisymmetric pattern and carries
    the full 2*pi azimuthal weight.

    The NumPy path keeps the historical ``np.gradient`` theta weights and
    multiplication order. ``jax_trapezoid=True`` keeps the optimization
    objective's differentiable trapezoid rule over theta and applies the phi
    weights as a constant of the intensity's dtype. Angles are static NumPy
    arrays in that path; intensity may be traced.
    """
    theta = np.asarray(theta)
    phi = np.asarray(phi)
    tol = _coverage_tol(theta, phi)
    theta_ok = (
        theta.ndim == 1 and theta.size >= 2 and np.all(np.isfinite(theta))
        and np.all(np.diff(theta) > 0)
        # Over-coverage beyond the poles: unchanged half-step limit.
        and theta[0] >= -(theta[1] - theta[0]) / 2 - tol
        and theta[-1] - np.pi <= (theta[-1] - theta[-2]) / 2 + tol
    )
    if theta_ok:
        if jax_trapezoid:
            lo_edge, hi_edge = theta[0], theta[-1]
        else:
            lo_edge = theta[0] - (theta[1] - theta[0]) / 2
            hi_edge = theta[-1] + (theta[-1] - theta[-2]) / 2
        theta_ok = (lo_edge <= POLE_TOLERANCE_RAD
                    and np.pi - hi_edge <= POLE_TOLERANCE_RAD)
    if not theta_ok:
        raise _coverage_error()
    dph = _phi_weights(phi)

    if jax_trapezoid:
        import jax.numpy as jnp

        sin_theta = jnp.asarray(np.sin(theta), dtype=intensity.dtype)
        integrand = intensity * sin_theta[None, :, None]
        power_phi = jnp.trapezoid(integrand, theta, axis=1)
        weights = jnp.asarray(dph, dtype=intensity.dtype)
        return jnp.sum(power_phi * weights[None, :], axis=1)

    dth = np.gradient(theta)
    if phi.size > 1 and np.issubdtype(phi.dtype, np.floating):
        # phi's own dtype, as np.gradient(phi) had it; one sample keeps the
        # historical float64 2*pi.
        dph = dph.astype(phi.dtype)
    integrand = intensity * np.sin(theta)[None, :, None]
    return np.sum(integrand * dth[None, :, None] * dph[None, None, :],
                  axis=(1, 2))
