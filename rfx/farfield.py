"""Near-to-far-field transform for antenna radiation patterns.

Uses the surface equivalence principle: record tangential E and H on a
closed Huygens box during simulation (via running DFT), then compute
far-field radiation integrals.

The DFT accumulation runs inside ``jax.lax.scan`` for efficiency.
Far-field post-processing uses NumPy (runs once after simulation).

References:
    Taflove & Hagness, "Computational Electrodynamics", 3rd ed., Ch. 8
    Balanis, "Advanced Engineering Electromagnetics", Ch. 12
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from rfx.grid import Grid, C0
from rfx.core.yee import EPS_0, MU_0

ETA_0 = float(np.sqrt(MU_0 / EPS_0))  # ~377 ohm


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class NTFFBox(NamedTuple):
    """Near-to-far-field transform surface (Huygens box).

    Grid indices defining a closed rectangular box. Must be inside the
    computational domain and outside any PEC or source regions.
    """
    i_lo: int
    i_hi: int
    j_lo: int
    j_hi: int
    k_lo: int
    k_hi: int
    freqs: jnp.ndarray  # (n_freqs,) Hz


class NTFFData(NamedTuple):
    """Accumulated DFT of tangential fields on 6 faces.

    x faces store [ey, ez, hy, hz], y faces [ex, ez, hx, hz],
    z faces [ex, ey, hx, hy].  Shape: (n_freqs, face_n1, face_n2, 4).
    """
    x_lo: jnp.ndarray
    x_hi: jnp.ndarray
    y_lo: jnp.ndarray
    y_hi: jnp.ndarray
    z_lo: jnp.ndarray
    z_hi: jnp.ndarray


class FarFieldResult(NamedTuple):
    """Far-field radiation result.

    E_theta, E_phi : (n_freqs, n_theta, n_phi) complex
        Angular far-field components (V·m, omitting 1/r factor).
    theta : (n_theta,) radians
    phi : (n_phi,) radians
    freqs : (n_freqs,) Hz
    """
    E_theta: np.ndarray
    E_phi: np.ndarray
    theta: np.ndarray
    phi: np.ndarray
    freqs: np.ndarray


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

def make_ntff_box(
    grid: Grid,
    corner_lo: tuple[float, float, float],
    corner_hi: tuple[float, float, float],
    freqs,
) -> NTFFBox:
    """Create an NTFF box from physical coordinates."""
    lo = grid.position_to_index(corner_lo)
    hi = grid.position_to_index(corner_hi)
    return NTFFBox(
        i_lo=lo[0], i_hi=hi[0],
        j_lo=lo[1], j_hi=hi[1],
        k_lo=lo[2], k_hi=hi[2],
        freqs=jnp.asarray(freqs, dtype=jnp.float32),
    )


def init_ntff_data(box: NTFFBox) -> NTFFData:
    """Initialize zeroed NTFF DFT accumulators.

    Uses complex128 (float64 real/imag) to avoid catastrophic cancellation
    during long DFT accumulations over thousands of timesteps.  At coarse
    dx (e.g., 2 mm at 5 GHz), the phase rotation in each step is small
    (dt ~ 4 ps), and float32 accumulation can lose the signal entirely.

    Automatically enables JAX float64 support if not already enabled.
    """
    import jax
    if not jax.config.x64_enabled:
        jax.config.update("jax_enable_x64", True)
        import warnings
        warnings.warn(
            "NTFF requires float64 precision. Enabled JAX x64 mode "
            "automatically. Set JAX_ENABLE_X64=1 to suppress this warning.",
            stacklevel=2,
        )

    nf = len(box.freqs)
    ni = box.i_hi - box.i_lo
    nj = box.j_hi - box.j_lo
    nk = box.k_hi - box.k_lo
    return NTFFData(
        x_lo=jnp.zeros((nf, nj, nk, 4), dtype=jnp.complex128),
        x_hi=jnp.zeros((nf, nj, nk, 4), dtype=jnp.complex128),
        y_lo=jnp.zeros((nf, ni, nk, 4), dtype=jnp.complex128),
        y_hi=jnp.zeros((nf, ni, nk, 4), dtype=jnp.complex128),
        z_lo=jnp.zeros((nf, ni, nj, 4), dtype=jnp.complex128),
        z_hi=jnp.zeros((nf, ni, nj, 4), dtype=jnp.complex128),
    )


# ---------------------------------------------------------------------------
# DFT accumulation (runs inside jax.lax.scan)
# ---------------------------------------------------------------------------

def accumulate_ntff(
    ntff_data: NTFFData,
    state,
    box: NTFFBox,
    dt: float,
    step_idx,
) -> NTFFData:
    """Accumulate one timestep of tangential field DFTs on all 6 faces.

    Called from the scan body.  ``step_idx`` comes from the scan xs.
    """
    t = jnp.float64(step_idx) * jnp.float64(dt)
    freqs_64 = jnp.asarray(box.freqs, dtype=jnp.float64)
    phase = jnp.exp(-1j * 2 * jnp.pi * freqs_64 * t) * dt  # (nf,) complex128
    ph = phase[:, None, None, None]  # broadcast to (nf, n1, n2, 4)

    i0, i1 = box.i_lo, box.i_hi
    j0, j1 = box.j_lo, box.j_hi
    k0, k1 = box.k_lo, box.k_hi

    def _x_face(idx):
        return jnp.stack([
            state.ey[idx, j0:j1, k0:k1],
            state.ez[idx, j0:j1, k0:k1],
            state.hy[idx, j0:j1, k0:k1],
            state.hz[idx, j0:j1, k0:k1],
        ], axis=-1)  # (nj, nk, 4)

    def _y_face(idx):
        return jnp.stack([
            state.ex[i0:i1, idx, k0:k1],
            state.ez[i0:i1, idx, k0:k1],
            state.hx[i0:i1, idx, k0:k1],
            state.hz[i0:i1, idx, k0:k1],
        ], axis=-1)

    def _z_face(idx):
        return jnp.stack([
            state.ex[i0:i1, j0:j1, idx],
            state.ey[i0:i1, j0:j1, idx],
            state.hx[i0:i1, j0:j1, idx],
            state.hy[i0:i1, j0:j1, idx],
        ], axis=-1)

    return NTFFData(
        x_lo=ntff_data.x_lo + ph * _x_face(i0)[None],
        x_hi=ntff_data.x_hi + ph * _x_face(i1)[None],
        y_lo=ntff_data.y_lo + ph * _y_face(j0)[None],
        y_hi=ntff_data.y_hi + ph * _y_face(j1)[None],
        z_lo=ntff_data.z_lo + ph * _z_face(k0)[None],
        z_hi=ntff_data.z_hi + ph * _z_face(k1)[None],
    )


# ---------------------------------------------------------------------------
# Far-field computation (post-simulation, NumPy)
# ---------------------------------------------------------------------------

def _surface_currents(fields, axis, sign):
    """Compute J_s = n x H, M_s = -n x E from stored tangential DFTs.

    Parameters
    ----------
    fields : (..., 4) complex — stored tangential components
    axis : 0, 1, 2 — face normal axis
    sign : +1 (hi face) or -1 (lo face)

    Returns (J, M) each (..., 3) in (x, y, z) order.
    """
    s = sign
    f0, f1, f2, f3 = (fields[..., i] for i in range(4))
    z = np.zeros_like(f0)

    if axis == 0:  # [ey, ez, hy, hz]
        J = np.stack([z, -s * f3, s * f2], axis=-1)
        M = np.stack([z, s * f1, -s * f0], axis=-1)
    elif axis == 1:  # [ex, ez, hx, hz]
        J = np.stack([s * f3, z, -s * f2], axis=-1)
        M = np.stack([-s * f1, z, s * f0], axis=-1)
    else:  # [ex, ey, hx, hy]
        J = np.stack([-s * f3, s * f2, z], axis=-1)
        M = np.stack([s * f1, -s * f0, z], axis=-1)

    return J, M


def _face_positions(axis, idx, other_ranges, dx, cpml):
    """Build (n1, n2, 3) position array for a face."""
    x_fixed = (idx - cpml) * dx

    if axis == 0:
        j_range, k_range = other_ranges
        y = (np.arange(j_range[0], j_range[1]) - cpml) * dx
        z = (np.arange(k_range[0], k_range[1]) - cpml) * dx
        Y, Z = np.meshgrid(y, z, indexing="ij")
        X = np.full_like(Y, x_fixed)
    elif axis == 1:
        i_range, k_range = other_ranges
        x = (np.arange(i_range[0], i_range[1]) - cpml) * dx
        z = (np.arange(k_range[0], k_range[1]) - cpml) * dx
        X, Z = np.meshgrid(x, z, indexing="ij")
        Y = np.full_like(X, x_fixed)
    else:
        i_range, j_range = other_ranges
        x = (np.arange(i_range[0], i_range[1]) - cpml) * dx
        y = (np.arange(j_range[0], j_range[1]) - cpml) * dx
        X, Y = np.meshgrid(x, y, indexing="ij")
        Z = np.full_like(X, x_fixed)

    return np.stack([X, Y, Z], axis=-1)  # (n1, n2, 3)


def compute_far_field(
    ntff_data: NTFFData,
    box: NTFFBox,
    grid: Grid,
    theta: np.ndarray,
    phi: np.ndarray,
) -> FarFieldResult:
    """Compute far-field radiation pattern from NTFF DFT data.

    Automatically dispatches to the JAX-differentiable implementation
    when called inside ``jax.grad`` or other JAX tracing contexts.
    Use this function for both post-processing and optimization objectives.

    Parameters
    ----------
    ntff_data : NTFFData
        Accumulated frequency-domain tangential fields.
    box : NTFFBox
    grid : Grid
    theta : (n_theta,) array in radians [0, pi]
    phi : (n_phi,) array in radians [0, 2*pi]

    Returns
    -------
    FarFieldResult
    """
    # Auto-detect JAX tracing context and dispatch to differentiable version
    import jax
    try:
        if any(isinstance(getattr(ntff_data, f, None), jax.core.Tracer)
               for f in ('x_lo', 'x_hi', 'y_lo')):
            return compute_far_field_jax(ntff_data, box, grid, theta, phi)
    except Exception:
        pass
    theta = np.asarray(theta, dtype=np.float64)
    phi = np.asarray(phi, dtype=np.float64)
    freqs = np.asarray(box.freqs, dtype=np.float64)
    nf = len(freqs)
    k_arr = 2 * np.pi * freqs / C0  # (nf,)

    dx = grid.dx
    dS = dx * dx
    cpml = grid.cpml_layers
    i0, i1 = box.i_lo, box.i_hi
    j0, j1 = box.j_lo, box.j_hi
    k0, k1 = box.k_lo, box.k_hi

    # Observation direction unit vectors
    TH, PH = np.meshgrid(theta, phi, indexing="ij")
    sth, cth = np.sin(TH), np.cos(TH)
    sph, cph = np.sin(PH), np.cos(PH)

    r_hat = np.stack([sth * cph, sth * sph, cth], axis=-1)     # (nθ, nφ, 3)
    th_hat = np.stack([cth * cph, cth * sph, -sth], axis=-1)
    ph_hat = np.stack([-sph, cph, np.zeros_like(sth)], axis=-1)

    n_th, n_ph = len(theta), len(phi)
    r_flat = r_hat.reshape(-1, 3)      # (n_dir, 3)
    n_dir = r_flat.shape[0]

    N_total = np.zeros((nf, n_dir, 3), dtype=np.complex128)
    L_total = np.zeros((nf, n_dir, 3), dtype=np.complex128)

    # Process each face
    face_specs = [
        # (data, axis, sign, face_idx, other_ranges)
        (ntff_data.x_lo, 0, -1, i0, ((j0, j1), (k0, k1))),
        (ntff_data.x_hi, 0, +1, i1, ((j0, j1), (k0, k1))),
        (ntff_data.y_lo, 1, -1, j0, ((i0, i1), (k0, k1))),
        (ntff_data.y_hi, 1, +1, j1, ((i0, i1), (k0, k1))),
        (ntff_data.z_lo, 2, -1, k0, ((i0, i1), (j0, j1))),
        (ntff_data.z_hi, 2, +1, k1, ((i0, i1), (j0, j1))),
    ]

    for face_dft, axis, sign, face_idx, other_ranges in face_specs:
        face_np = np.asarray(face_dft, dtype=np.complex128)  # (nf, n1, n2, 4)
        n1, n2 = face_np.shape[1], face_np.shape[2]
        if n1 == 0 or n2 == 0:
            continue

        pos = _face_positions(axis, face_idx, other_ranges, dx, cpml)
        pos_flat = pos.reshape(-1, 3)     # (nc, 3)
        fields_flat = face_np.reshape(nf, -1, 4)  # (nf, nc, 4)

        J, M = _surface_currents(fields_flat, axis, sign)  # (nf, nc, 3)

        dot = r_flat @ pos_flat.T  # (n_dir, nc)

        for fi in range(nf):
            phase = np.exp(1j * k_arr[fi] * dot)  # (n_dir, nc)
            N_total[fi] += np.einsum("dc,cj->dj", phase, J[fi]) * dS
            L_total[fi] += np.einsum("dc,cj->dj", phase, M[fi]) * dS

    # Project onto theta and phi unit vectors
    th_flat = th_hat.reshape(-1, 3)
    ph_flat = ph_hat.reshape(-1, 3)

    N_th = np.sum(N_total * th_flat[None, :, :], axis=-1)  # (nf, n_dir)
    N_ph = np.sum(N_total * ph_flat[None, :, :], axis=-1)
    L_th = np.sum(L_total * th_flat[None, :, :], axis=-1)
    L_ph = np.sum(L_total * ph_flat[None, :, :], axis=-1)

    # Far-field components
    E_theta = np.zeros((nf, n_dir), dtype=np.complex128)
    E_phi = np.zeros((nf, n_dir), dtype=np.complex128)
    for fi in range(nf):
        jk = 1j * k_arr[fi]
        E_theta[fi] = -jk / (4 * np.pi) * (L_ph[fi] + ETA_0 * N_th[fi])
        E_phi[fi] = jk / (4 * np.pi) * (L_th[fi] - ETA_0 * N_ph[fi])

    return FarFieldResult(
        E_theta=E_theta.reshape(nf, n_th, n_ph),
        E_phi=E_phi.reshape(nf, n_th, n_ph),
        theta=theta,
        phi=phi,
        freqs=freqs,
    )


# ---------------------------------------------------------------------------
# JAX-differentiable far-field (for use inside optimize / jax.grad)
# ---------------------------------------------------------------------------

def _surface_currents_jax(fields, axis, sign):
    """JAX version of _surface_currents."""
    f0, f1, f2, f3 = (fields[..., i] for i in range(4))
    z = jnp.zeros_like(f0)
    s = sign
    if axis == 0:
        J = jnp.stack([z, -s * f3, s * f2], axis=-1)
        M = jnp.stack([z, s * f1, -s * f0], axis=-1)
    elif axis == 1:
        J = jnp.stack([s * f3, z, -s * f2], axis=-1)
        M = jnp.stack([-s * f1, z, s * f0], axis=-1)
    else:
        J = jnp.stack([-s * f3, s * f2, z], axis=-1)
        M = jnp.stack([s * f1, -s * f0, z], axis=-1)
    return J, M


def _face_positions_jax(axis, idx, other_ranges, dx, cpml):
    """JAX version of _face_positions."""
    x_fixed = (idx - cpml) * dx
    if axis == 0:
        j_range, k_range = other_ranges
        y = (jnp.arange(j_range[0], j_range[1]) - cpml) * dx
        z = (jnp.arange(k_range[0], k_range[1]) - cpml) * dx
        Y, Z = jnp.meshgrid(y, z, indexing="ij")
        X = jnp.full_like(Y, x_fixed)
    elif axis == 1:
        i_range, k_range = other_ranges
        x = (jnp.arange(i_range[0], i_range[1]) - cpml) * dx
        z = (jnp.arange(k_range[0], k_range[1]) - cpml) * dx
        X, Z = jnp.meshgrid(x, z, indexing="ij")
        Y = jnp.full_like(X, x_fixed)
    else:
        i_range, j_range = other_ranges
        x = (jnp.arange(i_range[0], i_range[1]) - cpml) * dx
        y = (jnp.arange(j_range[0], j_range[1]) - cpml) * dx
        X, Y = jnp.meshgrid(x, y, indexing="ij")
        Z = jnp.full_like(X, x_fixed)
    return jnp.stack([X, Y, Z], axis=-1)


def compute_far_field_jax(
    ntff_data,
    box,
    grid,
    theta,
    phi,
):
    """JAX-differentiable far-field computation for use inside jax.grad.

    Same physics as ``compute_far_field`` but uses ``jnp`` throughout,
    enabling end-to-end differentiation for far-field optimization
    objectives (e.g., beam steering, gain maximization).

    Parameters
    ----------
    ntff_data : NTFFData (JAX arrays from lax.scan accumulation)
    box : NTFFBox
    grid : Grid
    theta : array in radians
    phi : array in radians

    Returns
    -------
    FarFieldResult with JAX arrays
    """
    theta = jnp.asarray(theta, dtype=jnp.float32)
    phi = jnp.asarray(phi, dtype=jnp.float32)
    freqs = jnp.asarray(box.freqs, dtype=jnp.float32)
    nf = freqs.shape[0]
    k_arr = 2 * jnp.pi * freqs / C0

    dx = grid.dx
    dS = dx * dx
    cpml = grid.cpml_layers
    i0, i1 = box.i_lo, box.i_hi
    j0, j1 = box.j_lo, box.j_hi
    k0, k1 = box.k_lo, box.k_hi

    TH, PH = jnp.meshgrid(theta, phi, indexing="ij")
    sth, cth = jnp.sin(TH), jnp.cos(TH)
    sph, cph = jnp.sin(PH), jnp.cos(PH)

    r_hat = jnp.stack([sth * cph, sth * sph, cth], axis=-1)
    th_hat = jnp.stack([cth * cph, cth * sph, -sth], axis=-1)
    ph_hat = jnp.stack([-sph, cph, jnp.zeros_like(sth)], axis=-1)

    n_th, n_ph = theta.shape[0], phi.shape[0]
    r_flat = r_hat.reshape(-1, 3)
    n_dir = r_flat.shape[0]

    N_total = jnp.zeros((nf, n_dir, 3), dtype=jnp.complex64)
    L_total = jnp.zeros((nf, n_dir, 3), dtype=jnp.complex64)

    face_specs = [
        (ntff_data.x_lo, 0, -1, i0, ((j0, j1), (k0, k1))),
        (ntff_data.x_hi, 0, +1, i1, ((j0, j1), (k0, k1))),
        (ntff_data.y_lo, 1, -1, j0, ((i0, i1), (k0, k1))),
        (ntff_data.y_hi, 1, +1, j1, ((i0, i1), (k0, k1))),
        (ntff_data.z_lo, 2, -1, k0, ((i0, i1), (j0, j1))),
        (ntff_data.z_hi, 2, +1, k1, ((i0, i1), (j0, j1))),
    ]

    for face_dft, axis, sign, face_idx, other_ranges in face_specs:
        # Keep DFT precision from accumulation (complex128 if available)
        face = jnp.asarray(face_dft)
        n1, n2 = face.shape[1], face.shape[2]
        if n1 == 0 or n2 == 0:
            continue

        pos = _face_positions_jax(axis, face_idx, other_ranges, dx, cpml)
        pos_flat = pos.reshape(-1, 3)
        fields_flat = face.reshape(nf, -1, 4)

        J, M = _surface_currents_jax(fields_flat, axis, sign)

        dot = r_flat @ pos_flat.T  # (n_dir, nc)

        # Vectorized over frequency (no Python loop)
        phase = jnp.exp(1j * k_arr[:, None, None] * dot[None, :, :])  # (nf, n_dir, nc)
        N_total = N_total + jnp.einsum("fdc,fcj->fdj", phase, J) * dS
        L_total = L_total + jnp.einsum("fdc,fcj->fdj", phase, M) * dS

    th_flat = th_hat.reshape(-1, 3)
    ph_flat = ph_hat.reshape(-1, 3)

    N_th = jnp.sum(N_total * th_flat[None, :, :], axis=-1)
    N_ph = jnp.sum(N_total * ph_flat[None, :, :], axis=-1)
    L_th = jnp.sum(L_total * th_flat[None, :, :], axis=-1)
    L_ph = jnp.sum(L_total * ph_flat[None, :, :], axis=-1)

    jk = 1j * k_arr[:, None]  # (nf, 1)
    E_theta = -jk / (4 * jnp.pi) * (L_ph + ETA_0 * N_th)
    E_phi = jk / (4 * jnp.pi) * (L_th - ETA_0 * N_ph)

    return FarFieldResult(
        E_theta=E_theta.reshape(nf, n_th, n_ph),
        E_phi=E_phi.reshape(nf, n_th, n_ph),
        theta=theta,
        phi=phi,
        freqs=freqs,
    )


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------

def radiation_pattern(ff: FarFieldResult) -> np.ndarray:
    """Normalized radiation pattern in dB.

    Returns (n_freqs, n_theta, n_phi) array.
    """
    power = np.abs(ff.E_theta) ** 2 + np.abs(ff.E_phi) ** 2
    peak = np.max(power, axis=(1, 2), keepdims=True)
    safe_peak = np.where(peak > 0, peak, 1.0)
    return 10 * np.log10(np.maximum(power / safe_peak, 1e-10))


def directivity(ff: FarFieldResult) -> np.ndarray:
    """Directivity in dBi for each frequency.

    Integrates radiated power over the sphere using trapezoidal rule
    and computes D = 4π U_max / P_rad.

    Returns (n_freqs,) array.
    """
    power = np.abs(ff.E_theta) ** 2 + np.abs(ff.E_phi) ** 2  # (nf, nθ, nφ)
    theta = ff.theta
    dth = np.gradient(theta) if len(theta) > 1 else np.array([np.pi])
    dph = np.gradient(ff.phi) if len(ff.phi) > 1 else np.array([2 * np.pi])

    sin_th = np.sin(theta)  # (nθ,)
    # Integrate: P_rad = ∫∫ U sin(θ) dθ dφ
    integrand = power * sin_th[None, :, None]  # (nf, nθ, nφ)
    P_rad = np.sum(integrand * dth[None, :, None] * dph[None, None, :], axis=(1, 2))

    U_max = np.max(power, axis=(1, 2))
    safe_P = np.where(P_rad > 0, P_rad, 1.0)

    D = 4 * np.pi * U_max / safe_P
    return 10 * np.log10(np.maximum(D, 1e-10))


def axial_ratio(ff: FarFieldResult) -> np.ndarray:
    """Axial ratio (AR) of far-field polarization.

    AR = |E_major| / |E_minor| (≥ 1). AR = 1 for circular, ∞ for linear.

    Returns (n_freqs, n_theta, n_phi) array.
    """
    E_th = ff.E_theta
    E_ph = ff.E_phi

    # Polarization ellipse from E_theta and E_phi (complex phasors)
    # Semi-major and semi-minor from eigenvalues of the coherency matrix
    a2 = np.abs(E_th)**2
    b2 = np.abs(E_ph)**2
    c = E_th * np.conj(E_ph)

    # Stokes parameters
    S0 = a2 + b2
    S1 = a2 - b2
    S3 = 2 * np.imag(c)

    # Axial ratio from Stokes
    np.sqrt(S1**2 + S3**2)
    safe_S0 = np.where(S0 > 0, S0, 1.0)

    # sin(2χ) = S3/S0 where χ is ellipticity angle
    sin2chi = np.clip(S3 / safe_S0, -1.0, 1.0)
    chi = 0.5 * np.arcsin(sin2chi)

    # AR = |1/tan(χ)| (cot of ellipticity angle)
    tan_chi = np.tan(chi)
    safe_tan = np.where(np.abs(tan_chi) > 1e-10, tan_chi, 1e-10)
    AR = np.abs(1.0 / safe_tan)
    AR = np.minimum(AR, 1000.0)  # cap at 1000 (essentially linear)
    return AR


def axial_ratio_dB(ff: FarFieldResult) -> np.ndarray:
    """Axial ratio in dB. 0 dB = circular, large = linear."""
    return 20 * np.log10(axial_ratio(ff))


def polarization_tilt(ff: FarFieldResult) -> np.ndarray:
    """Polarization tilt angle (orientation of major axis) in radians.

    Returns (n_freqs, n_theta, n_phi) array.
    """
    E_th = ff.E_theta
    E_ph = ff.E_phi

    a2 = np.abs(E_th)**2
    b2 = np.abs(E_ph)**2
    c = E_th * np.conj(E_ph)

    S1 = a2 - b2
    S2 = 2 * np.real(c)

    # Tilt angle τ = 0.5 * atan2(S2, S1)
    return 0.5 * np.arctan2(S2, S1)


def polarization_sense(ff: FarFieldResult) -> np.ndarray:
    """Polarization sense: +1 = RHCP, -1 = LHCP, 0 = linear.

    Returns (n_freqs, n_theta, n_phi) array of integers.
    """
    E_th = ff.E_theta
    E_ph = ff.E_phi
    S3 = 2 * np.imag(E_th * np.conj(E_ph))
    return np.sign(S3).astype(int)
