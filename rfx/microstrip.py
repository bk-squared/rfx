"""Closed-form microstrip transmission-line synthesis and analysis.

Host-side (numpy/math) analytical helpers for the single most common RF
design task: *what trace width gives a target characteristic impedance on
a given substrate?* These are pure closed-form functions — they do **not**
run a simulation and are intentionally decoupled from JAX.

Formula source
--------------
Characteristic impedance and effective permittivity use the
**Hammerstad–Jensen** quasi-static synthesis/analysis equations as
presented in:

    E. Hammerstad and O. Jensen, "Accurate Models for Microstrip
    Computer-Aided Design," IEEE MTT-S International Microwave Symposium
    Digest, 1980, pp. 407-409.

and reproduced in standard references (e.g. Pozar, *Microwave
Engineering*, 4th ed., §3.8; Wadell, *Transmission Line Design
Handbook*). The effective-permittivity term uses the Hammerstad form

    eps_eff = (eps_r + 1)/2 + (eps_r - 1)/2 * (1 + 12 h / w)^(-1/2)

with the thin-line ``(w/h < 1)`` correction term, and the impedance is
the standard two-regime Wheeler/Hammerstad expression. Width synthesis
uses the closed-form Wheeler inversion (Pozar eqs. 3.197).

Accuracy
--------
The quasi-static Hammerstad–Jensen model is accurate to roughly **1 %**
for ``0.05 <= w/h <= 20`` and ``eps_r <= ~13`` at low frequency. It
ignores conductor thickness, dispersion (frequency dependence of
``eps_eff``), and surface roughness, so it is a starting-point synthesis
tool — not a substitute for a full-wave solve. All lengths are in metres.
"""

from __future__ import annotations

import math
import warnings

# Free-space wave impedance, eta_0 = sqrt(mu_0 / eps_0)  (ohm).
_ETA_0 = 376.730313668


def _validate_geometry(height: float, eps_r: float, *, width: float | None = None) -> None:
    """Validate common microstrip inputs, raising ValueError on bad input."""
    if width is not None and not (width > 0.0):
        raise ValueError(f"width must be positive (metres), got {width!r}")
    if not (height > 0.0):
        raise ValueError(f"height must be positive (metres), got {height!r}")
    if not (eps_r >= 1.0):
        raise ValueError(f"eps_r must be >= 1.0, got {eps_r!r}")


def microstrip_eps_eff(width: float, height: float, eps_r: float) -> float:
    """Effective relative permittivity of a microstrip line (Hammerstad).

    Parameters
    ----------
    width : float
        Conductor (trace) width in metres.
    height : float
        Substrate (dielectric) thickness in metres.
    eps_r : float
        Substrate relative permittivity (dimensionless, >= 1).

    Returns
    -------
    float
        Effective relative permittivity ``eps_eff`` seen by the quasi-TEM
        mode (``1 <= eps_eff <= eps_r``).

    Notes
    -----
    Uses the Hammerstad form (Pozar §3.8, eq. 3.195); the
    ``(w/h < 1)`` branch adds the ``(1 - w/h)^2`` thin-line correction.
    """
    _validate_geometry(height, eps_r, width=width)
    u = width / height  # w/h ratio
    if u >= 1.0:
        f = (1.0 + 12.0 / u) ** -0.5
    else:
        f = (1.0 + 12.0 / u) ** -0.5 + 0.04 * (1.0 - u) ** 2
    return (eps_r + 1.0) / 2.0 + (eps_r - 1.0) / 2.0 * f


def microstrip_impedance(
    width: float, height: float, eps_r: float
) -> tuple[float, float]:
    """Characteristic impedance and eps_eff of a microstrip line (analysis).

    Parameters
    ----------
    width : float
        Conductor (trace) width in metres.
    height : float
        Substrate (dielectric) thickness in metres.
    eps_r : float
        Substrate relative permittivity (dimensionless, >= 1).

    Returns
    -------
    (z0, eps_eff) : tuple of float
        ``z0`` is the characteristic impedance in ohms; ``eps_eff`` is the
        effective relative permittivity.

    Notes
    -----
    Two-regime Wheeler/Hammerstad expression (Pozar §3.8, eq. 3.196):

    * ``w/h <= 1``: ``z0 = eta_0 / (2*pi*sqrt(eps_eff))
      * ln(8 h/w + w/(4 h))``
    * ``w/h >= 1``: ``z0 = eta_0 / (sqrt(eps_eff)
      * (w/h + 1.393 + 0.667 ln(w/h + 1.444)))``
    """
    _validate_geometry(height, eps_r, width=width)
    eps_eff = microstrip_eps_eff(width, height, eps_r)
    u = width / height
    sqrt_eff = math.sqrt(eps_eff)
    if u <= 1.0:
        z0 = _ETA_0 / (2.0 * math.pi * sqrt_eff) * math.log(8.0 / u + u / 4.0)
    else:
        z0 = _ETA_0 / (sqrt_eff * (u + 1.393 + 0.667 * math.log(u + 1.444)))
    return z0, eps_eff


def microstrip_width(z0: float, height: float, eps_r: float) -> float:
    """Trace width that yields a target characteristic impedance (synthesis).

    Solves the inverse problem: *given* a desired ``z0`` (e.g. 50 ohm), a
    substrate thickness, and a substrate permittivity, return the
    microstrip conductor width.

    Parameters
    ----------
    z0 : float
        Target characteristic impedance in ohms (> 0).
    height : float
        Substrate (dielectric) thickness in metres.
    eps_r : float
        Substrate relative permittivity (dimensionless, >= 1).

    Returns
    -------
    float
        Conductor (trace) width in metres.

    Notes
    -----
    Closed-form Wheeler/Hammerstad synthesis (Pozar §3.8, eq. 3.197).
    A trial ``w/h`` is computed for both the narrow and wide regimes and
    the branch consistent with its own validity condition is selected.
    """
    if not (z0 > 0.0):
        raise ValueError(f"z0 must be positive (ohm), got {z0!r}")
    _validate_geometry(height, eps_r)

    # Wide-strip trial (w/h > 2): B-based form. Its logarithms are only real
    # for b > 1 (low-to-moderate z0); for high-impedance lines on high-eps_r
    # substrates b <= 1 and the wide form does not apply — mark it invalid and
    # fall through to the narrow branch instead of raising a bare
    # "math domain error".
    b = _ETA_0 * math.pi / (2.0 * z0 * math.sqrt(eps_r))
    if b > 1.0:
        u_wide = (2.0 / math.pi) * (
            b - 1.0 - math.log(2.0 * b - 1.0)
            + (eps_r - 1.0) / (2.0 * eps_r)
            * (math.log(b - 1.0) + 0.39 - 0.61 / eps_r)
        )
    else:
        u_wide = float("-inf")  # wide form invalid here; use the narrow branch

    # Narrow-strip trial (w/h < 2): A-based form.
    a = (
        z0 / 60.0 * math.sqrt((eps_r + 1.0) / 2.0)
        + (eps_r - 1.0) / (eps_r + 1.0) * (0.23 + 0.11 / eps_r)
    )
    u_narrow = 8.0 * math.exp(a) / (math.exp(2.0 * a) - 2.0)

    # Each closed form is valid on one side of w/h = 2: use the wide form when
    # its own trial comes out >= 2, otherwise the narrow form (Pozar rule).
    u = u_wide if u_wide >= 2.0 else u_narrow

    # The Hammerstad-Jensen fit is only ~1% accurate for 0.05 <= w/h <= 20;
    # outside that the returned width is an extrapolation — warn rather than
    # silently hand back a degraded value.
    if not (0.05 <= u <= 20.0):
        warnings.warn(
            f"microstrip_width: resulting w/h={u:.3g} is outside the "
            f"Hammerstad-Jensen validated range [0.05, 20]; the returned "
            f"width is an extrapolation with degraded accuracy "
            f"(z0={z0!r} ohm, eps_r={eps_r!r}).",
            stacklevel=2,
        )

    return u * height
