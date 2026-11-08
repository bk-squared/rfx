"""Line impedance from a coax source's realized, unit-voltage TEM mode."""

from __future__ import annotations

import numpy as np

from rfx.core.yee import EPS_0
from rfx.grid import C0, Grid


def coaxial_built_line_impedance(
    grid: Grid, ex_profile: np.ndarray, ey_profile: np.ndarray, eps_r: float,
) -> float:
    """Return Z of the homogeneous line on the source's stamped PEC mask.

    Pass the solved unit-voltage Ex/Ey edge profiles, before source amplitude
    scaling. On the uniform Yee grid each transverse edge has dual area dx².
    The energy per length gives C' = eps0 eps_r sum(E² A_edge); TEM then gives
    Z = 1 / (v C'), v = c0 / sqrt(eps_r). Shorted edges carry zero field.
    No declared annulus area or radius enters this energy integral.
    """
    ex, ey = np.asarray(ex_profile, dtype=float), np.asarray(ey_profile, dtype=float)
    if ex.shape != grid.shape[:2] or ey.shape != ex.shape:
        raise ValueError("coax TEM profiles must match the transverse grid shape")
    if not np.isfinite(eps_r) or eps_r <= 0:
        raise ValueError("coax TEM fill permittivity must be finite and positive")
    energy_sum = float(np.sum(ex * ex) + np.sum(ey * ey)) * float(grid.dx) ** 2
    if not np.isfinite(energy_sum) or energy_sum <= 0:
        raise ValueError("coax TEM profile must have finite positive stored energy")
    capacitance_per_m = EPS_0 * float(eps_r) * energy_sum
    return float(np.sqrt(eps_r) / (C0 * capacitance_per_m))
