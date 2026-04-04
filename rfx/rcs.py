"""Radar Cross Section (RCS) computation pipeline.

Combines TFSF plane-wave illumination with NTFF near-to-far-field
transform to compute monostatic and bistatic RCS of scatterers.

The standard FDTD RCS approach:
1. TFSF illuminates the target with a plane wave
2. Scattered field (outside TFSF box) is captured by NTFF box
3. NTFF computes far-field pattern
4. RCS(theta, phi) = 4*pi*r^2 * |E_scat|^2 / |E_inc|^2

Reference: Taflove & Hagness, Ch. 8-9.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from rfx.grid import Grid
from rfx.core.yee import MaterialArrays
from rfx.farfield import (
    NTFFBox, compute_far_field,
)
from rfx.sources.tfsf import init_tfsf
from rfx.simulation import run


class RCSResult(NamedTuple):
    """Radar cross section computation result.

    freqs : (n_freqs,) Hz
    theta : (n_theta,) radians
    phi : (n_phi,) radians
    rcs_dbsm : (n_freqs, n_theta, n_phi) in dBsm
    rcs_linear : (n_freqs, n_theta, n_phi) in m^2
    monostatic_rcs : (n_freqs,) backscatter RCS in dBsm, or None
    """
    freqs: np.ndarray
    theta: np.ndarray
    phi: np.ndarray
    rcs_dbsm: np.ndarray
    rcs_linear: np.ndarray
    monostatic_rcs: np.ndarray | None


def _incident_spectrum_amplitude(
    f0: float,
    bandwidth: float,
    freqs: np.ndarray,
    dt: float,
    n_steps: int,
) -> np.ndarray:
    """Compute the frequency-domain amplitude of the TFSF incident pulse.

    The TFSF 1-D source uses a differentiated Gaussian:
        s(t) = -2*arg * exp(-arg^2),  arg = (t - t0) / tau
    where tau = 1/(f0 * bandwidth * pi), t0 = 3*tau.

    We compute the DFT of this waveform at the requested frequencies
    to get the incident field spectral amplitude for RCS normalization.
    """
    tau = 1.0 / (f0 * bandwidth * np.pi)
    t0 = 3.0 * tau
    times = np.arange(n_steps) * dt
    arg = (times - t0) / tau
    waveform = -2.0 * arg * np.exp(-(arg ** 2))

    # DFT at requested frequencies
    # S(f) = sum_n s(n*dt) * exp(-j*2*pi*f*n*dt) * dt
    amplitudes = np.zeros(len(freqs), dtype=np.complex128)
    for i, f in enumerate(freqs):
        phase = np.exp(-1j * 2 * np.pi * f * times)
        amplitudes[i] = np.sum(waveform * phase) * dt

    return amplitudes


def compute_rcs(
    grid: Grid,
    materials: MaterialArrays,
    n_steps: int,
    *,
    f0: float,
    bandwidth: float = 0.5,
    theta_inc: float = 0.0,
    phi_inc: float = 0.0,
    polarization: str = "ez",
    theta_obs: jnp.ndarray | np.ndarray | None = None,
    phi_obs: jnp.ndarray | np.ndarray | None = None,
    freqs: jnp.ndarray | np.ndarray | None = None,
    boundary: str = "cpml",
    cpml_layers: int = 8,
    tfsf_margin: int = 3,
    ntff_offset: int = 1,
) -> RCSResult:
    """Compute radar cross section of the scatterer defined in materials.

    Parameters
    ----------
    grid : Grid
        Simulation grid (must already include CPML padding).
    materials : MaterialArrays
        Material arrays with the scatterer defined (e.g., PEC regions
        with very high conductivity or high eps_r).
    n_steps : int
        Number of FDTD timesteps.
    f0 : float
        Center frequency of the Gaussian pulse (Hz).
    bandwidth : float
        Fractional bandwidth of the pulse.
    theta_inc : float
        Incident angle in degrees (0 = +x propagation). Currently only
        normal incidence (0) is supported.
    phi_inc : float
        Incident azimuth in degrees (reserved for future oblique support).
    polarization : str
        Electric field polarization: "ez" or "ey".
    theta_obs : array or None
        Observation elevation angles in radians. Default: 0 to pi, 37 points.
    phi_obs : array or None
        Observation azimuth angles in radians. Default: [0, pi/2].
    freqs : array or None
        Frequencies at which to compute RCS (Hz). Default: [f0].
    boundary : str
        Boundary condition type ("cpml" or "pec").
    cpml_layers : int
        Number of CPML layers (must match grid.cpml_layers).
    tfsf_margin : int
        Cells between CPML edge and TFSF boundary.
    ntff_offset : int
        Cells between TFSF boundary and NTFF box (NTFF must be in the
        scattered-field region, i.e., outside the TFSF box).

    Returns
    -------
    RCSResult
        freqs, theta, phi, rcs_dbsm (dBsm), rcs_linear (m^2),
        monostatic_rcs (dBsm at backscatter direction).
    """
    # Defaults
    if theta_obs is None:
        theta_obs = np.linspace(0.01, np.pi - 0.01, 37)
    else:
        theta_obs = np.asarray(theta_obs, dtype=np.float64)

    if phi_obs is None:
        phi_obs = np.array([0.0, np.pi / 2])
    else:
        phi_obs = np.asarray(phi_obs, dtype=np.float64)

    if freqs is None:
        freqs_arr = np.array([f0], dtype=np.float64)
    else:
        freqs_arr = np.asarray(freqs, dtype=np.float64)

    len(freqs_arr)
    dx = grid.dx
    dt = grid.dt

    # --- 1. Set up TFSF source ---
    tfsf_cfg, tfsf_st = init_tfsf(
        nx=grid.nx,
        dx=dx,
        dt=dt,
        cpml_layers=cpml_layers,
        tfsf_margin=tfsf_margin,
        f0=f0,
        bandwidth=bandwidth,
        amplitude=1.0,
        polarization=polarization,
        direction="+x",
        angle_deg=theta_inc,
    )

    # --- 2. Set up NTFF box just outside TFSF box ---
    # NTFF box must be in scattered-field region (outside TFSF box).
    # Place it `ntff_offset` cells outside the TFSF boundaries.
    ntff_i_lo = tfsf_cfg.x_lo - ntff_offset
    ntff_i_hi = tfsf_cfg.x_hi + ntff_offset + 1
    ntff_j_lo = cpml_layers + ntff_offset
    ntff_j_hi = grid.ny - cpml_layers - ntff_offset
    ntff_k_lo = cpml_layers + ntff_offset
    ntff_k_hi = grid.nz - cpml_layers - ntff_offset

    # Clamp to valid range
    ntff_i_lo = max(ntff_i_lo, 1)
    ntff_i_hi = min(ntff_i_hi, grid.nx - 2)
    ntff_j_lo = max(ntff_j_lo, 1)
    ntff_j_hi = min(ntff_j_hi, grid.ny - 2)
    ntff_k_lo = max(ntff_k_lo, 1)
    ntff_k_hi = min(ntff_k_hi, grid.nz - 2)

    ntff_box = NTFFBox(
        i_lo=ntff_i_lo,
        i_hi=ntff_i_hi,
        j_lo=ntff_j_lo,
        j_hi=ntff_j_hi,
        k_lo=ntff_k_lo,
        k_hi=ntff_k_hi,
        freqs=jnp.array(freqs_arr, dtype=jnp.float32),
    )

    # --- 3. Run simulation with TFSF + NTFF ---
    result = run(
        grid,
        materials,
        n_steps,
        boundary=boundary,
        tfsf=(tfsf_cfg, tfsf_st),
        ntff=ntff_box,
    )

    # --- 4. Compute far-field from NTFF data ---
    ff = compute_far_field(
        result.ntff_data,
        ntff_box,
        grid,
        theta_obs,
        phi_obs,
    )

    # --- 5. Compute incident field spectrum for normalization ---
    E_inc_spectrum = _incident_spectrum_amplitude(
        f0, bandwidth, freqs_arr, dt, n_steps,
    )

    # --- 6. Compute RCS ---
    # RCS = 4*pi * |E_far|^2 / |E_inc|^2
    # where E_far already includes the jk/(4*pi) factor from compute_far_field,
    # so: |E_far|^2 = |E_theta|^2 + |E_phi|^2
    # The far-field result has E_theta, E_phi in V*m (omitting 1/r).
    # The NTFF formulation gives: E_far(r) = (jk/4*pi*r) * [N, L integrals]
    # So |E_far * r|^2 = |E_theta|^2 + |E_phi|^2 as returned.
    #
    # RCS = 4*pi * r^2 * |E_scat|^2 / |E_inc|^2
    #     = 4*pi * |E_far_r|^2 / |E_inc|^2
    # where E_far_r = r * E_far (the quantity returned by compute_far_field).

    E_theta = np.asarray(ff.E_theta, dtype=np.complex128)  # (nf, n_theta, n_phi)
    E_phi = np.asarray(ff.E_phi, dtype=np.complex128)

    power_scat = np.abs(E_theta) ** 2 + np.abs(E_phi) ** 2  # (nf, n_theta, n_phi)
    power_inc = np.abs(E_inc_spectrum) ** 2  # (nf,)

    # Avoid division by zero
    safe_power_inc = np.where(power_inc > 0, power_inc, 1e-30)

    rcs_linear = 4.0 * np.pi * power_scat / safe_power_inc[:, None, None]

    # Convert to dBsm
    rcs_dbsm = 10.0 * np.log10(np.maximum(rcs_linear, 1e-30))

    # --- 7. Extract monostatic (backscatter) RCS ---
    # For +x incidence, backscatter direction is theta=pi, phi=0
    # (i.e., -x direction in spherical coordinates).
    # Find the closest observation angle to (theta=pi, phi=0).
    monostatic_rcs = None
    if len(theta_obs) > 0 and len(phi_obs) > 0:
        # Backscatter: theta = pi (or close to it), phi = 0 (or close)
        theta_back_idx = np.argmin(np.abs(theta_obs - np.pi))
        phi_back_idx = np.argmin(np.abs(phi_obs - 0.0))
        mono_linear = rcs_linear[:, theta_back_idx, phi_back_idx]
        monostatic_rcs = 10.0 * np.log10(np.maximum(mono_linear, 1e-30))

    return RCSResult(
        freqs=freqs_arr,
        theta=theta_obs,
        phi=phi_obs,
        rcs_dbsm=rcs_dbsm,
        rcs_linear=rcs_linear,
        monostatic_rcs=monostatic_rcs,
    )
