"""Shared full-sphere power integration, independent of far-field containers."""

import numpy as np


def integrate_radiated_power(intensity, theta, phi, *, jax_trapezoid=False):
    """Integrate ``intensity * sin(theta)`` over solid angle per frequency.

    ``intensity`` has shape (n_freqs, n_theta, n_phi). Angles must be finite,
    increasing vectors: theta reaches 0 and pi within half its respective
    endpoint step, and phi spans a full turn within one endpoint step (the
    larger of the first and last steps). A single phi sample assumes an
    axisymmetric pattern and carries the full 2*pi azimuthal weight.

    Keep the historical NumPy gradient weights and multiplication order.
    ``jax_trapezoid=True`` retains the optimization objective's differentiable
    theta-then-phi trapezoid rule instead, including its endpoint weights.
    Angles are static NumPy arrays in that path; intensity may be traced.
    """
    theta = np.asarray(theta)
    phi = np.asarray(phi)
    # Permit rounding at the half-step / one-step coverage boundaries only.
    tol = 8 * np.finfo(np.result_type(theta.dtype, phi.dtype, np.float32)).eps * np.pi
    theta_ok = (
        theta.ndim == 1 and theta.size >= 2 and np.all(np.isfinite(theta))
        and np.all(np.diff(theta) > 0)
        and abs(theta[0]) <= (theta[1] - theta[0]) / 2 + tol
        and abs(theta[-1] - np.pi) <= (theta[-1] - theta[-2]) / 2 + tol
    )
    phi_ok = (
        phi.ndim == 1 and phi.size >= 1 and np.all(np.isfinite(phi))
        and (phi.size == 1 or (
            np.all(np.diff(phi) > 0)
            and abs(phi[-1] - phi[0] - 2 * np.pi)
            <= max(phi[1] - phi[0], phi[-1] - phi[-2]) + tol
        ))
    )
    if not (theta_ok and phi_ok):
        raise ValueError(
            "Total radiated power needs full-sphere coverage: theta must span "
            "[0, pi] with endpoints within a half-step, and phi must span a "
            "full turn within one step or have exactly one sample "
            "(axisymmetric assumption). Angles must be finite and increasing. "
            "Realized gain with input_power needs no full-sphere coverage."
        )

    if jax_trapezoid:
        import jax.numpy as jnp

        sin_theta = jnp.asarray(np.sin(theta), dtype=intensity.dtype)
        integrand = intensity * sin_theta[None, :, None]
        power_phi = jnp.trapezoid(integrand, theta, axis=1)
        if len(phi) == 1:
            return power_phi[:, 0] * (2 * np.pi)
        return jnp.trapezoid(power_phi, phi, axis=1)

    dth = np.gradient(theta)
    dph = np.gradient(phi) if len(phi) > 1 else np.array([2 * np.pi])
    integrand = intensity * np.sin(theta)[None, :, None]
    return np.sum(integrand * dth[None, :, None] * dph[None, None, :],
                  axis=(1, 2))
