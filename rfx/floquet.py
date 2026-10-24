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

import jax.numpy as jnp
import numpy as np

from rfx.grid import C0
from rfx.core.yee import FDTDState, EPS_0, MU_0


# ---------------------------------------------------------------------------
# Phase shift computation
# ---------------------------------------------------------------------------



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

    Only the specular ``(0,0)`` mode with **TE** polarization is implemented (the
    extractor reads the (Ex, Hy) tangential pair and the TE wave impedance
    ``eta0/cos(theta)``). ``n_modes > 1`` (higher-order grating lobes) and TM
    polarization are NOT implemented and are rejected fail-loud rather than
    silently returning wrong results (RF-audit 2026-07-23).

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
    if int(n_modes) != 1:
        raise NotImplementedError(
            f"extract_floquet_modes only implements the specular (0,0) mode "
            f"(n_modes=1); got n_modes={n_modes}. Higher-order Floquet grating "
            f"lobes are not yet extracted — this fails loud instead of silently "
            f"returning only the (0,0) mode."
        )

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

    .. warning:: **Experimental.** This pipeline carries an E0 contract test
        (``tests/unit/ports/test_floquet_s_params_contract.py``) — the first real pytest
        caller, a passivity check on pre-isolated pure-wave input, and a
        finite-grad AD smoke (the path is jnp-pure and AD-traceable) — but **no
        accuracy validation against an external periodic-cell solver**. It also
        presumes the input accumulators already isolate the incident /
        reflected / transmitted waves: a naive single-run soft-source drive
        does NOT (a bidirectional source leaves no pure-forward monitor plane),
        so ``|S11|`` from such a drive can exceed 1. Do not build accuracy
        claims on it. Tracking: https://github.com/bk-squared/rfx/issues/141

        **Drive-isolation scoping (2026-07-05, R2-STOP):** the obvious fix — a
        two-run empty-vs-device reference subtraction (``E_scat = E_dev −
        E_empty`` at a +z-side plane, the pattern that works for waveguide/coax
        ports) — does NOT resolve this for a periodic Floquet cell. On an εr=4
        slab vs the exact analytic Airy S11, the two-run extraction is
        non-physical: |S11|≈1.5 at band-center 6 GHz where Airy=0.09 (the ~1.5
        extraction-error floor swamps the small true reflection) and larger
        blow-ups at band edges (small-incident normalization noise). TFSF
        one-way plane-wave sources — the clean isolation route — are explicitly
        barred with Floquet ports (see ``add_floquet_port``). A real fix needs
        dedicated numerics (clean incident normalization + higher-order-mode
        filtering, or a Floquet-compatible one-way source), NOT a parameter
        flip; and drive-isolation is the PREREQUISITE to any external (RCWA /
        Meep-Bloch) accuracy validation — the external reference is moot until
        FDTD yields a physical |S11|≤1 on a real periodic structure.

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


# Lazy compatibility exports keep record definitions free of import cycles.
_LAZY_EXPORTS = {
    "floquet_phase_shift": "rfx.measurement.floquet",
    "inject_floquet_source": "rfx.stepping.floquet",
    "update_floquet_dft": "rfx.measurement.floquet",
}


def __getattr__(name):
    from importlib import import_module

    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(_LAZY_EXPORTS[name]), name)
    globals()[name] = value
    return value
