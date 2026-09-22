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

__all__ = ["slab_te0_neff", "measure_neff_two_wave", "TwoWaveFit"]


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


class TwoWaveFit(tuple):
    """``(n_eff, beta, rel_residual, b_over_a)`` for one frequency."""
    __slots__ = ()

    def __new__(cls, n_eff, beta, rel_residual, b_over_a):
        return tuple.__new__(cls, (float(n_eff), float(beta),
                                   float(rel_residual), float(b_over_a)))

    n_eff = property(lambda self: self[0])
    beta = property(lambda self: self[1])
    rel_residual = property(lambda self: self[2])
    b_over_a = property(lambda self: self[3])


def _two_wave_residual(beta, x, y):
    """Relative LS residual of ``y ~ A e^{-i beta x} + B e^{+i beta x}``."""
    design = np.stack([np.exp(-1j * beta * x), np.exp(1j * beta * x)], axis=1)
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    resid = y - design @ coef
    return float(np.linalg.norm(resid) / np.linalg.norm(y)), coef


def measure_neff_two_wave(field_line, x, freqs_hz, *, c0: float,
                          eps_core: float, eps_clad: float,
                          n_scan: int = 4001, n_refine: int = 80):
    """Recover ``n_eff(f)`` from a field line that carries BOTH directions.

    A lossless uniform section supports exactly two propagating solutions of one
    bound mode, ``A exp(-i beta x)`` and ``B exp(+i beta x)``.  This fits that
    model -- ``beta`` by a 1-D search, ``A`` and ``B`` linearly at each trial
    ``beta`` -- instead of assuming the line is a single travelling wave.

    Written for crossval 03 (#812), where a single-mode linear-phase fit was
    falsified by its own residual self-check: that guide carries ``|B/A| ~ 0.53``
    and a phase-slope estimator reads it as a +/-0.7 % wobble in ``n_eff``.  The
    two-wave fit is not a repair of a noisy estimator; it is the model the
    physics actually has.

    Parameters
    ----------
    field_line : (n_freqs, n_x) complex array
        DFT phasors of one field component along the propagation axis.
    x : (n_x,) float array
        Sample positions, metres.
    freqs_hz : (n_freqs,) float array
    c0 : float
        Speed of light, same unit system.
    eps_core, eps_clad : float
        Permittivities that set the search bracket.  A bound mode has
        ``sqrt(eps_clad) < n_eff < sqrt(eps_core)`` -- the window between the
        cladding and core light lines.  This is a first-principles bracket; it
        is not derived from any measurement.
    n_scan : int
        Coarse samples across the bracket (global minimum).
    n_refine : int
        Bisection steps of the local refinement.

    Returns
    -------
    list[TwoWaveFit]
        One per frequency.  ``rel_residual`` is the witness that the two-wave
        premise held; ``b_over_a`` is the standing-wave ratio, REPORTED, not
        gated -- its correct value is a property of the boundary treatment, not
        of this comparator.
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
    if x.size < 8:
        raise ValueError("need at least 8 samples for a two-wave fit")
    if not (eps_core > eps_clad > 0.0):
        raise ValueError("need eps_core > eps_clad > 0 for a bound-mode bracket")

    n_lo = np.sqrt(eps_clad)
    n_hi = np.sqrt(eps_core)
    out = []
    for i, f in enumerate(freqs_hz):
        y = field_line[i]
        k0 = 2.0 * np.pi * f / c0
        lo, hi = k0 * n_lo * (1.0 + 1e-6), k0 * n_hi * (1.0 - 1e-6)
        betas = np.linspace(lo, hi, n_scan)
        costs = [_two_wave_residual(b, x, y)[0] for b in betas]
        j = int(np.argmin(costs))
        step = betas[1] - betas[0]
        lo_r = max(lo, betas[j] - step)
        hi_r = min(hi, betas[j] + step)
        for _ in range(n_refine):
            mid = 0.5 * (lo_r + hi_r)
            h = 0.25 * (hi_r - lo_r)
            if _two_wave_residual(mid - h, x, y)[0] <= \
                    _two_wave_residual(mid + h, x, y)[0]:
                hi_r = mid + h
            else:
                lo_r = mid - h
        beta = 0.5 * (lo_r + hi_r)
        rel, coef = _two_wave_residual(beta, x, y)
        b_over_a = float(np.abs(coef[1]) / np.abs(coef[0])) if coef[0] != 0 \
            else float("inf")
        out.append(TwoWaveFit(beta / k0, beta, rel, b_over_a))
    return out
