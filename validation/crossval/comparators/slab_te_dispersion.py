"""Closed-form symmetric-slab TE dispersion oracle + a phase-fit n_eff estimator.

Written for crossval case 03 (#812): the case's only pass gate was
``T = flux_out/flux_in = 1``, an identity of the flux bookkeeping that holds for
any bound mode in any lossless uniform section, so it could not see the guide.
This module supplies the quantity that *does* depend on the guide.

Two halves, deliberately independent of each other and of the FDTD run:

``slab_te0_neff``
    The oracle.  Closed-form fundamental even TE mode of a symmetric dielectric
    slab.  No FDTD input of any kind; it is a function of the declared
    permittivities, the declared thickness, and the frequency.

``measure_neff_from_line``
    The estimator.  Recovers ``beta`` from the spatial phase of a complex field
    line taken along the propagation axis.  It never sees the oracle.

Polarization, stated once so it cannot drift: rfx ``mode="2d_tmz"`` carries
(Ez, Hx, Hy) with the slab infinite in x and z and bounded in y, so ``Ez`` lies
*in* the slab faces.  That is the slab-waveguide **TE** mode.  Ez then obeys the
scalar Helmholtz equation with Ez and dEz/dy continuous across each interface,
which gives the even-mode eigenvalue equation ``u tan u = w`` used below.  (The
orthogonal convention -- H parallel to the faces -- would give ``u tan u =
(n1/n2)^2 w`` and a different number; it is not this case's polarization.)
"""

from __future__ import annotations

import numpy as np

__all__ = ["slab_te0_neff", "measure_neff_from_line", "PhaseFit"]


def slab_te0_neff(eps_core: float, eps_clad: float, thickness: float,
                  k0: float) -> float:
    """Effective index of the fundamental even TE mode of a symmetric slab.

    Parameters
    ----------
    eps_core, eps_clad : float
        Relative permittivities.  ``eps_core > eps_clad`` is required for a
        bound mode to exist.
    thickness : float
        Slab thickness ``d``, in the same length unit as ``1/k0``.
    k0 : float
        Free-space wavenumber ``2*pi*f/c``, in the reciprocal of that unit.

    Returns
    -------
    float
        ``n_eff = beta / k0``.  Lies strictly in ``(sqrt(eps_clad),
        sqrt(eps_core))``.

    Notes
    -----
    Eigenvalue equation for the even TE mode (Ez cosine in the core, decaying
    exponential in the cladding, Ez and dEz/dy continuous)::

        u tan(u) = w,   u^2 + w^2 = V^2,   V = (k0 d / 2) sqrt(n1^2 - n2^2)
        kappa = 2u/d,   beta = sqrt(n1^2 k0^2 - kappa^2)

    The fundamental root always exists for any V > 0 (the slab TE0 mode has no
    cutoff) and lies in ``(0, min(V, pi/2))``, so a plain bisection on a
    bracketed, monotone-crossing residual is exact to machine precision and
    needs no initial guess.
    """
    if not (eps_core > eps_clad > 0.0):
        raise ValueError(
            f"need eps_core > eps_clad > 0 for a bound slab mode, got "
            f"eps_core={eps_core}, eps_clad={eps_clad}")
    if thickness <= 0.0 or k0 <= 0.0:
        raise ValueError(f"thickness and k0 must be positive, got "
                         f"thickness={thickness}, k0={k0}")

    n1 = float(np.sqrt(eps_core))
    n2 = float(np.sqrt(eps_clad))
    V = 0.5 * k0 * thickness * np.sqrt(n1 * n1 - n2 * n2)

    def residual(u: float) -> float:
        return u * np.tan(u) - np.sqrt(max(V * V - u * u, 0.0))

    # residual(0+) = -V < 0; residual(u -> min(V, pi/2)-) > 0.
    hi = min(V, 0.5 * np.pi)
    lo, hi = 1e-15, hi * (1.0 - 1e-14)
    if residual(hi) <= 0.0:          # V < pi/2: the root is at u -> V
        hi = V * (1.0 - 1e-15)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if residual(mid) < 0.0:
            lo = mid
        else:
            hi = mid
    u = 0.5 * (lo + hi)

    kappa = 2.0 * u / thickness
    beta_sq = n1 * n1 * k0 * k0 - kappa * kappa
    if beta_sq <= 0.0:                                   # pragma: no cover
        raise RuntimeError("no bound TE0 root found")
    return float(np.sqrt(beta_sq) / k0)


class PhaseFit(tuple):
    """``(n_eff, beta, residual_rms_rad)`` for one frequency."""
    __slots__ = ()

    def __new__(cls, n_eff, beta, residual_rms):
        return tuple.__new__(cls, (float(n_eff), float(beta),
                                   float(residual_rms)))

    n_eff = property(lambda self: self[0])
    beta = property(lambda self: self[1])
    residual_rms = property(lambda self: self[2])


def measure_neff_from_line(field_line, x, freqs_hz, *, c0: float,
                           forward: str = "+x"):
    """Recover ``n_eff(f)`` from the spatial phase of a complex field line.

    Parameters
    ----------
    field_line : (n_freqs, n_x) complex array
        DFT phasors of one field component sampled along the propagation axis.
    x : (n_x,) float array
        Sample positions, metres, monotonically increasing.
    freqs_hz : (n_freqs,) float array
        Frequencies, Hz.
    c0 : float
        Speed of light in the same unit system.
    forward : "+x" or "-x"
        Propagation direction of the wave being measured.

    Returns
    -------
    list[PhaseFit]
        One entry per frequency.

    Notes
    -----
    The probe accumulates ``X(f) = sum_t x(t) exp(-i 2 pi f t) dt``, so a wave
    ``cos(omega t - beta x)`` has ``arg X = -beta x``: the phase slope is
    ``-beta`` for +x propagation.  The unwrap is unambiguous only while
    ``beta * dx < pi``; the caller is responsible for the sampling, and this
    function raises if the *measured* per-sample phase step reaches pi.

    The residual RMS of the linear fit is returned rather than discarded: it is
    the witness that the single-mode linear-phase premise actually held.  A
    contaminated line (two modes, a standing wave, radiation) shows it directly.
    """
    field_line = np.asarray(field_line)
    x = np.asarray(x, dtype=float)
    freqs_hz = np.asarray(freqs_hz, dtype=float)
    if field_line.ndim != 2 or field_line.shape[1] != x.size:
        raise ValueError(
            f"field_line must be (n_freqs, n_x) matching x; got "
            f"{field_line.shape} vs n_x={x.size}")
    if field_line.shape[0] != freqs_hz.size:
        raise ValueError("field_line rows must match freqs_hz")
    if x.size < 4:
        raise ValueError("need at least 4 samples for a phase fit")
    if forward not in ("+x", "-x"):
        raise ValueError(f"forward must be '+x' or '-x', got {forward!r}")

    sign = -1.0 if forward == "+x" else +1.0
    out = []
    for i, f in enumerate(freqs_hz):
        phi = np.unwrap(np.angle(field_line[i]))
        step = np.max(np.abs(np.diff(phi)))
        if step >= np.pi:
            raise ValueError(
                f"phase step {step:.3f} rad >= pi at f={f:.4e} Hz: the sample "
                "spacing does not resolve the guided wavelength, so the "
                "unwrap is ambiguous")
        slope, intercept = np.polyfit(x, phi, 1)
        beta = sign * slope
        resid = phi - (slope * x + intercept)
        k0 = 2.0 * np.pi * f / c0
        out.append(PhaseFit(beta / k0, beta,
                            float(np.sqrt(np.mean(resid ** 2)))))
    return out
