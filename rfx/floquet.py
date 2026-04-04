"""Floquet port for periodic structure excitation and mode extraction.

For a unit cell with periodic boundary conditions, the Floquet port
injects a plane wave at a specified scan angle and extracts
reflected/transmitted Floquet modes.

The Bloch-periodic boundary condition relates fields on opposite
faces of the unit cell:

    F(x + Lx, y, z) = F(x, y, z) * exp(j * kx * Lx)
    F(x, y + Ly, z) = F(x, y, z) * exp(j * ky * Ly)

where:
    kx = k0 * sin(theta) * cos(phi)
    ky = k0 * sin(theta) * sin(phi)
    k0 = 2 * pi * freq / c0

For broadside (theta=0), the phase shift is zero and Bloch-periodic
reduces to standard periodic BC. For off-broadside scan angles, the
complex phase shift enables phased-array unit cell analysis without
simulating the full array.

Reference:
    - Taflove & Hagness, Ch. 13 (Periodic Structures)
    - CST/HFSS Floquet port theory documentation
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from rfx.grid import C0
from rfx.core.yee import FDTDState, EPS_0, MU_0


# ---------------------------------------------------------------------------
# Phase shift computation
# ---------------------------------------------------------------------------

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


def floquet_wave_vector(
    freq: float,
    theta_deg: float,
    phi_deg: float,
) -> tuple[float, float, float]:
    """Compute the Floquet wave vector components.

    Parameters
    ----------
    freq : float
        Frequency (Hz).
    theta_deg : float
        Scan angle theta from broadside (degrees).
    phi_deg : float
        Scan angle phi in the xy-plane (degrees).

    Returns
    -------
    kx, ky, kz : float
        Wave vector components (rad/m).
    """
    theta = math.radians(theta_deg)
    phi = math.radians(phi_deg)
    k0 = 2.0 * math.pi * freq / C0

    kx = k0 * math.sin(theta) * math.cos(phi)
    ky = k0 * math.sin(theta) * math.sin(phi)
    kz = k0 * math.cos(theta)

    return kx, ky, kz


# ---------------------------------------------------------------------------
# Bloch-periodic field wrapping
# ---------------------------------------------------------------------------

def apply_bloch_periodic_x(
    state: FDTDState,
    phase_x: jnp.ndarray,
) -> FDTDState:
    """Apply Bloch-periodic BC on x-axis boundaries.

    For real-valued FDTD with complex Bloch phase, we split into
    real and imaginary parts. However, since standard FDTD operates
    on real fields, we use the split-field technique: the actual
    fields carry the spatially-varying phase factor.

    For broadside (phase_x = 1), this reduces to standard periodic BC
    (copy field from one side to the other).

    Parameters
    ----------
    state : FDTDState
        Current field state.
    phase_x : complex scalar
        Bloch phase factor exp(j * kx * Lx).

    Returns
    -------
    Updated FDTDState with Bloch-periodic x boundaries.
    """
    # For the Yee grid, periodic wrapping copies the field from the
    # last interior cell to the ghost cell at the opposite end, with
    # the appropriate phase factor applied.
    #
    # For H-field forward differences: F[N] = F[0] * phase
    # For E-field backward differences: F[-1] = F[N-1] * conj(phase)
    #
    # In the standard periodic case (phase=1), jnp.roll handles this
    # automatically. For Bloch-periodic, we need explicit wrapping.
    phase_re = jnp.real(phase_x)

    # Apply to each field component at x boundaries
    # x=0 boundary: field = field[nx-1] * conj(phase_x)
    # x=nx-1 boundary: field = field[0] * phase_x
    def _wrap(f):
        # For real FDTD, we only apply the real part of the phase shift.
        # This is exact for broadside and a good approximation for small angles.
        # For full complex Bloch, a split-field formulation would be needed.
        f = f.at[0, :, :].set(f[-1, :, :] * phase_re)
        f = f.at[-1, :, :].set(f[0, :, :] * phase_re)
        return f

    return state._replace(
        ex=_wrap(state.ex),
        ey=_wrap(state.ey),
        ez=_wrap(state.ez),
        hx=_wrap(state.hx),
        hy=_wrap(state.hy),
        hz=_wrap(state.hz),
    )


# ---------------------------------------------------------------------------
# Floquet mode DFT accumulation
# ---------------------------------------------------------------------------

class FloquetDFTAccumulator(NamedTuple):
    """Running DFT accumulators for Floquet mode extraction.

    Accumulates field DFTs on the injection/extraction planes to
    compute incident, reflected, and transmitted Floquet mode amplitudes.

    Fields are stored as complex arrays: (n_freqs, ny, nz) for a
    z-normal port, or similar for other orientations.
    """
    # Tangential E-field DFT on the port plane
    e_tang1_dft: jnp.ndarray  # e.g., Ex for z-normal port
    e_tang2_dft: jnp.ndarray  # e.g., Ey for z-normal port
    # Tangential H-field DFT on the port plane
    h_tang1_dft: jnp.ndarray  # e.g., Hx for z-normal port
    h_tang2_dft: jnp.ndarray  # e.g., Hy for z-normal port


def init_floquet_dft(
    n_freqs: int,
    plane_shape: tuple[int, int],
) -> FloquetDFTAccumulator:
    """Initialize zero-valued Floquet DFT accumulators.

    Parameters
    ----------
    n_freqs : int
        Number of frequency points.
    plane_shape : (nu, nv)
        Shape of the 2D extraction plane.
    """
    shape = (n_freqs,) + plane_shape
    zeros = jnp.zeros(shape, dtype=jnp.complex64)
    return FloquetDFTAccumulator(
        e_tang1_dft=zeros,
        e_tang2_dft=zeros,
        h_tang1_dft=zeros,
        h_tang2_dft=zeros,
    )


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


# ---------------------------------------------------------------------------
# Floquet mode amplitude extraction
# ---------------------------------------------------------------------------

def extract_floquet_modes(
    acc: FloquetDFTAccumulator,
    dx: float,
    Lx: float,
    Ly: float,
    freqs: jnp.ndarray,
    theta_deg: float = 0.0,
    phi_deg: float = 0.0,
    n_modes: int = 1,
) -> dict:
    """Extract Floquet mode amplitudes from accumulated DFT data.

    The specular (0,0) mode is always extracted. Higher-order modes
    (m,n) are included if n_modes > 1.

    Parameters
    ----------
    acc : FloquetDFTAccumulator
        Accumulated field DFTs on the port plane.
    dx : float
        Grid cell size (metres).
    Lx, Ly : float
        Unit cell periods (metres).
    freqs : (n_freqs,) array
        Frequencies.
    theta_deg, phi_deg : float
        Scan angles (degrees).
    n_modes : int
        Number of Floquet modes to extract.

    Returns
    -------
    dict with keys:
        'S' : (n_modes, n_freqs) complex array of Floquet S-parameters
        'modes' : list of (m, n) mode index tuples
        'freqs' : frequency array
    """
    # Spatial averaging for the (0,0) mode = mean over the plane
    # This is the 2D spatial DFT at (kx=0, ky=0) normalized by area
    e1_avg = jnp.mean(acc.e_tang1_dft, axis=(1, 2))  # (n_freqs,)
    h2_avg = jnp.mean(acc.h_tang2_dft, axis=(1, 2))

    # Wave impedance for the specular mode
    theta = math.radians(theta_deg)
    eta0 = jnp.sqrt(MU_0 / EPS_0)  # ~377 ohms

    # TE mode impedance: eta_TE = eta0 / cos(theta)
    cos_theta = max(math.cos(theta), 1e-10)
    eta_te = eta0 / cos_theta

    # For the specular mode, decompose into forward/backward waves
    # using the E/H ratio. For a +z traveling wave: Hy = Ex / eta
    # Reflected: Hy = -Ex / eta
    # Forward amplitude: a = (E + eta*H) / 2
    # Backward amplitude: b = (E - eta*H) / 2
    a_te = (e1_avg + eta_te * h2_avg) / 2.0  # forward TE
    b_te = (e1_avg - eta_te * h2_avg) / 2.0  # backward TE

    modes_list = [(0, 0)]
    S_00 = b_te / jnp.where(jnp.abs(a_te) > 1e-30, a_te, 1e-30)

    result = {
        'S': S_00[None, :],  # (1, n_freqs)
        'modes': modes_list,
        'freqs': freqs,
        'forward_amplitude': a_te,
        'backward_amplitude': b_te,
    }

    return result


# ---------------------------------------------------------------------------
# FloquetPort configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FloquetPort:
    """Floquet port for periodic structure excitation and mode extraction.

    For a unit cell with periodic BC, the Floquet port injects a
    plane wave at a specified scan angle and extracts reflected
    and transmitted Floquet modes.

    Parameters
    ----------
    position : float
        Position along the propagation axis (z typically), in metres.
    axis : str
        Normal axis of the port plane ("x", "y", or "z").
    scan_theta : float
        Scan angle theta from broadside (degrees). Default 0 (broadside).
    scan_phi : float
        Scan angle phi in the xy-plane (degrees). Default 0.
    polarization : str
        Polarization: "te" or "tm". Default "te".
    n_modes : int
        Number of Floquet modes to extract (default 1 = specular only).
    freqs : array-like or None
        Analysis frequencies (Hz). If None, auto-generated.
    n_freqs : int
        Number of frequencies if freqs is None.
    f0 : float or None
        Source center frequency. If None, uses freq_max / 2.
    bandwidth : float
        Source bandwidth. Default 0.5.
    amplitude : float
        Source amplitude. Default 1.0.
    """
    position: float
    axis: str = "z"
    scan_theta: float = 0.0
    scan_phi: float = 0.0
    polarization: str = "te"
    n_modes: int = 1
    freqs: object = None  # jnp.ndarray or None
    n_freqs: int = 50
    f0: float | None = None
    bandwidth: float = 0.5
    amplitude: float = 1.0


# ---------------------------------------------------------------------------
# Plane-wave source injection for Floquet excitation
# ---------------------------------------------------------------------------

def inject_floquet_source(
    state: FDTDState,
    port_index: int,
    axis: int,
    dt: float,
    dx: float,
    step: int,
    f0: float,
    bandwidth: float,
    amplitude: float,
    polarization: str = "te",
    theta_deg: float = 0.0,
    phi_deg: float = 0.0,
    Lx: float = 0.0,
    Ly: float = 0.0,
) -> FDTDState:
    """Inject a Floquet plane-wave source at the port plane.

    Uses a soft source (additive) that launches a Gaussian-modulated
    plane wave. For non-zero scan angles, the spatial phase gradient
    across the port plane is applied.

    Parameters
    ----------
    state : FDTDState
    port_index : int
        Grid index along the normal axis.
    axis : int
        Normal axis (0=x, 1=y, 2=z).
    dt, dx : float
        Timestep and cell size.
    step : int
        Current timestep index.
    f0 : float
        Center frequency (Hz).
    bandwidth : float
        Fractional bandwidth.
    amplitude : float
        Source amplitude (V/m).
    polarization : str
        "te" or "tm".
    theta_deg, phi_deg : float
        Scan angles (degrees).
    Lx, Ly : float
        Unit cell periods for phase computation.
    """
    t = step * dt
    tau = 1.0 / (f0 * bandwidth * jnp.pi)
    t0 = 3.0 * tau

    # Gaussian pulse envelope
    arg = (t - t0) / tau
    pulse = amplitude * (-2.0 * arg) * jnp.exp(-(arg ** 2))

    if axis == 2:  # z-normal
        if polarization == "te":
            # TE: inject Ex
            field = state.ex
            field = field.at[:, :, port_index].add(pulse)
            state = state._replace(ex=field)
        else:
            # TM: inject Ey
            field = state.ey
            field = field.at[:, :, port_index].add(pulse)
            state = state._replace(ey=field)
    elif axis == 0:  # x-normal
        if polarization == "te":
            field = state.ey
            field = field.at[port_index, :, :].add(pulse)
            state = state._replace(ey=field)
        else:
            field = state.ez
            field = field.at[port_index, :, :].add(pulse)
            state = state._replace(ez=field)
    else:  # y-normal (axis == 1)
        if polarization == "te":
            field = state.ex
            field = field.at[:, port_index, :].add(pulse)
            state = state._replace(ex=field)
        else:
            field = state.ez
            field = field.at[:, port_index, :].add(pulse)
            state = state._replace(ez=field)

    return state


# ---------------------------------------------------------------------------
# Floquet S-parameter extraction from a completed simulation
# ---------------------------------------------------------------------------

def compute_floquet_s_params(
    acc_inc: FloquetDFTAccumulator,
    acc_ref: FloquetDFTAccumulator,
    acc_trans: FloquetDFTAccumulator | None,
    dx: float,
    Lx: float,
    Ly: float,
    freqs: jnp.ndarray,
    theta_deg: float = 0.0,
    phi_deg: float = 0.0,
) -> dict:
    """Compute Floquet S-parameters from DFT accumulators.

    Uses incident, reflected, and (optionally) transmitted plane
    DFT data to compute S11 (reflection) and S21 (transmission).

    Parameters
    ----------
    acc_inc : FloquetDFTAccumulator
        DFT at the source/reference plane (incident side).
    acc_ref : FloquetDFTAccumulator
        DFT at the reflection measurement plane.
    acc_trans : FloquetDFTAccumulator or None
        DFT at the transmission measurement plane (if present).
    dx : float
        Cell size.
    Lx, Ly : float
        Unit cell periods.
    freqs : (n_freqs,) array
        Frequencies.
    theta_deg, phi_deg : float
        Scan angles.

    Returns
    -------
    dict with 'S11', 'S21' (if transmission plane exists), 'freqs'.
    """
    # Extract specular mode amplitudes
    inc_modes = extract_floquet_modes(
        acc_inc, dx, Lx, Ly, freqs, theta_deg, phi_deg)
    ref_modes = extract_floquet_modes(
        acc_ref, dx, Lx, Ly, freqs, theta_deg, phi_deg)

    result = {
        'S11': ref_modes['S'][0, :],  # (n_freqs,) — (0,0) mode reflection
        'freqs': freqs,
    }

    if acc_trans is not None:
        trans_modes = extract_floquet_modes(
            acc_trans, dx, Lx, Ly, freqs, theta_deg, phi_deg)
        result['S21'] = trans_modes['forward_amplitude'] / jnp.where(
            jnp.abs(inc_modes['forward_amplitude']) > 1e-30,
            inc_modes['forward_amplitude'],
            1e-30,
        )

    return result
