"""Holland thin-wire subcell model for FDTD.

Thin wires (radius << dx) cannot be resolved by the Yee grid. The Holland
(1981) model modifies the effective permittivity and conductivity along the
wire to account for the sub-cell capacitance and resistance.

Reference: Holland & Simpson, IEEE TEMC 23(2), 88-97, 1981.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import jax.numpy as jnp

from rfx.grid import Grid
from rfx.core.yee import EPS_0


class ThinWire(NamedTuple):
    """Axis-aligned thin wire definition.

    Parameters
    ----------
    start : (x, y, z) in meters
    end : (x, y, z) in meters
    radius : wire radius in meters
    conductivity : wire conductivity in S/m (default: copper 5.8e7)
    """
    start: tuple[float, float, float]
    end: tuple[float, float, float]
    radius: float
    conductivity: float = 5.8e7


def compute_thin_wire_correction(
    grid: Grid,
    wire: ThinWire,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute effective eps_r and sigma corrections for a thin wire.

    Parameters
    ----------
    grid : Grid
    wire : ThinWire (must be axis-aligned)

    Returns
    -------
    (eps_r_correction, sigma_correction) : arrays of shape grid.shape
        Add these to the base material arrays along the wire cells.
    """
    dx = grid.dx
    r = wire.radius
    nx, ny, nz = grid.shape

    eps_corr = np.zeros((nx, ny, nz), dtype=np.float32)
    sigma_corr = np.zeros((nx, ny, nz), dtype=np.float32)

    # Determine wire axis
    s = np.array(wire.start)
    e = np.array(wire.end)
    diff = e - s

    # Find which axis the wire is along
    nonzero = np.abs(diff) > 1e-10
    if np.sum(nonzero) != 1:
        raise ValueError("ThinWire must be axis-aligned (only one coordinate changes)")

    axis = int(np.argmax(nonzero))
    pad = np.array(grid.axis_pads) if hasattr(grid, 'axis_pads') else np.zeros(3)

    # Wire position in transverse plane (grid indices)
    trans_axes = [i for i in range(3) if i != axis]
    t0_idx = int(round(s[trans_axes[0]] / dx)) + int(pad[trans_axes[0]])
    t1_idx = int(round(s[trans_axes[1]] / dx)) + int(pad[trans_axes[1]])

    # Wire extent along its axis
    lo = min(s[axis], e[axis])
    hi = max(s[axis], e[axis])
    lo_idx = int(round(lo / dx)) + int(pad[axis])
    hi_idx = int(round(hi / dx)) + int(pad[axis])

    # Holland correction factors
    # Effective permittivity: eps_eff = eps_0 / (2*pi*ln(dx/(2*r)))
    # This replaces the cell's eps along the wire axis
    ln_ratio = np.log(dx / (2 * r)) if dx > 2 * r else 0.1
    eps_eff_factor = 1.0 / (2 * np.pi * ln_ratio)

    # Effective conductivity: sigma_eff = sigma * pi * r^2 / dx^2
    sigma_eff = wire.conductivity * np.pi * r ** 2 / dx ** 2

    # Apply along wire cells
    for a_idx in range(lo_idx, hi_idx + 1):
        if axis == 0:
            if 0 <= a_idx < nx and 0 <= t0_idx < ny and 0 <= t1_idx < nz:
                eps_corr[a_idx, t0_idx, t1_idx] = eps_eff_factor
                sigma_corr[a_idx, t0_idx, t1_idx] = sigma_eff
        elif axis == 1:
            if 0 <= t0_idx < nx and 0 <= a_idx < ny and 0 <= t1_idx < nz:
                eps_corr[t0_idx, a_idx, t1_idx] = eps_eff_factor
                sigma_corr[t0_idx, a_idx, t1_idx] = sigma_eff
        else:
            if 0 <= t0_idx < nx and 0 <= t1_idx < ny and 0 <= a_idx < nz:
                eps_corr[t0_idx, t1_idx, a_idx] = eps_eff_factor
                sigma_corr[t0_idx, t1_idx, a_idx] = sigma_eff

    return jnp.array(eps_corr), jnp.array(sigma_corr)
