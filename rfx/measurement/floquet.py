"""Floquet phase shifts and frequency-domain accumulation."""

from __future__ import annotations

from rfx.core.yee import FDTDState
from rfx.floquet import FloquetDFTAccumulator
from rfx.grid import C0
import jax.numpy as jnp
import math
import numpy as np


def floquet_phase_shift(
    Lx: float,
    Ly: float,
    freq: float,
    theta_deg: float,
    phi_deg: float,
) -> tuple[complex, complex]:
    """Compute Bloch phase shifts for periodic BC at a given scan angle.

    Parameters
    ----------
    Lx : float
        Unit cell period in x (metres).
    Ly : float
        Unit cell period in y (metres).
    freq : float
        Frequency (Hz).
    theta_deg : float
        Scan angle theta from broadside (degrees).
    phi_deg : float
        Scan angle phi in the xy-plane (degrees).

    Returns
    -------
    phase_x, phase_y : complex
        Bloch phase factors: exp(j * kx * Lx), exp(j * ky * Ly).
        At broadside (theta=0), both are 1+0j.
    """
    theta = math.radians(theta_deg)
    phi = math.radians(phi_deg)
    k0 = 2.0 * math.pi * freq / C0

    kx = k0 * math.sin(theta) * math.cos(phi)
    ky = k0 * math.sin(theta) * math.sin(phi)

    phase_x = complex(np.exp(1j * kx * Lx))
    phase_y = complex(np.exp(1j * ky * Ly))

    return phase_x, phase_y


def update_floquet_dft(
    acc: FloquetDFTAccumulator,
    state: FDTDState,
    port_index: int,
    axis: int,
    freqs: jnp.ndarray,
    dt: float,
    step: int,
) -> FloquetDFTAccumulator:
    """Accumulate one timestep into the Floquet DFT.

    Extracts tangential fields at the port plane and applies running
    DFT at each frequency point.

    Parameters
    ----------
    acc : FloquetDFTAccumulator
        Current accumulator state.
    state : FDTDState
        Current FDTD field state.
    port_index : int
        Grid index along the port normal axis.
    axis : int
        Normal axis (0=x, 1=y, 2=z).
    freqs : (n_freqs,) array
        Frequencies for DFT.
    dt : float
        Timestep.
    step : int
        Current timestep index.
    """
    t = step * dt
    # DFT kernel: exp(-j * 2*pi*f*t) for each frequency
    phase = jnp.exp(-1j * 2.0 * jnp.pi * freqs * t)  # (n_freqs,)

    # Extract tangential field plane based on normal axis
    if axis == 2:  # z-normal
        sl = (slice(None), slice(None), port_index)
        e1 = state.ex[sl]  # (nx, ny)
        e2 = state.ey[sl]
        h1 = state.hx[sl]
        h2 = state.hy[sl]
    elif axis == 0:  # x-normal
        sl = (port_index, slice(None), slice(None))
        e1 = state.ey[sl]  # (ny, nz)
        e2 = state.ez[sl]
        h1 = state.hy[sl]
        h2 = state.hz[sl]
    else:  # y-normal
        sl = (slice(None), port_index, slice(None))
        e1 = state.ex[sl]  # (nx, nz)
        e2 = state.ez[sl]
        h1 = state.hx[sl]
        h2 = state.hz[sl]

    # Accumulate: outer product of phase vector with field plane
    # phase: (n_freqs,), field: (nu, nv) -> (n_freqs, nu, nv)
    e1_contrib = phase[:, None, None] * e1[None, :, :]
    e2_contrib = phase[:, None, None] * e2[None, :, :]
    h1_contrib = phase[:, None, None] * h1[None, :, :]
    h2_contrib = phase[:, None, None] * h2[None, :, :]

    return FloquetDFTAccumulator(
        e_tang1_dft=acc.e_tang1_dft + e1_contrib,
        e_tang2_dft=acc.e_tang2_dft + e2_contrib,
        h_tang1_dft=acc.h_tang1_dft + h1_contrib,
        h_tang2_dft=acc.h_tang2_dft + h2_contrib,
    )
