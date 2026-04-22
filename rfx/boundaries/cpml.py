"""Convolutional Perfectly Matched Layer (CPML) absorbing boundary.

Implements the stretched-coordinate PML formulation with auxiliary
differential equation (ADE) update. Supports CFS-CPML with κ stretching
for improved evanescent wave absorption.

Standard CPML:  s(ω) = 1 + σ/(jω)
CFS-CPML:       s(ω) = κ + σ/(α + jω)

When kappa_max=1.0 (default), the CFS-CPML reduces exactly to standard CPML.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from rfx.grid import Grid
from rfx.core.yee import _shift_fwd, _shift_bwd
from rfx.core.jax_utils import is_tracer


class CPMLParams(NamedTuple):
    """CPML profile parameters for one axis direction."""

    # Damping profile σ (n_layers,)
    sigma: jnp.ndarray
    # Frequency-shift parameter κ (n_layers,)
    kappa: jnp.ndarray
    # CFS parameter α (n_layers,)
    alpha: jnp.ndarray
    # Precomputed update coefficients
    b: jnp.ndarray  # exp(-(σ/κ + α) * dt/ε₀)
    c: jnp.ndarray  # σ * (b - 1) / (σ*κ + κ²*α)


class CPMLAxisParams(NamedTuple):
    """Per-face CPML profiles (6 faces) for anisotropic grids.

    T7 Phase 2 PR1 refactor (v1.7.1): promoted the ``x`` / ``y`` axis-level
    profiles to explicit ``x_lo`` / ``x_hi`` / ``y_lo`` / ``y_hi`` fields.
    The hi-face profile arrays are pre-flipped so the scan body no longer
    calls ``jnp.flip`` at every step.

    On uniform grids the six profiles are computed from the same
    ``(dt, dx)`` input so their (σ, κ, α) arrays are identical up to
    the hi-face flip. On non-uniform z the z-lo and z-hi faces already
    use independent cell sizes.

    Cell sizes are stored as Python floats (not traced) so they
    can be used inside JIT-compiled scan bodies.
    """
    x_lo: CPMLParams    # x-lo face
    x_hi: CPMLParams    # x-hi face (flipped relative to x_lo in the uniform case)
    y_lo: CPMLParams    # y-lo face
    y_hi: CPMLParams    # y-hi face (flipped relative to y_lo in the uniform case)
    z_lo: CPMLParams    # z-lo face (uses dz at lo boundary)
    z_hi: CPMLParams    # z-hi face (uses dz at hi boundary, already independent pre-PR1)
    dx_x_lo: float = 0.0   # x-lo boundary cell size
    dx_x_hi: float = 0.0   # x-hi boundary cell size
    dx_y_lo: float = 0.0   # y-lo boundary cell size
    dx_y_hi: float = 0.0   # y-hi boundary cell size
    dz_lo: float = 0.0     # z-lo boundary cell size
    dz_hi: float = 0.0     # z-hi boundary cell size


class CPMLState(NamedTuple):
    """Auxiliary CPML fields (psi) for all 6 faces."""

    # Psi arrays for E-field update (from H-curl correction)
    psi_ex_ylo: jnp.ndarray
    psi_ex_yhi: jnp.ndarray
    psi_ex_zlo: jnp.ndarray
    psi_ex_zhi: jnp.ndarray
    psi_ey_xlo: jnp.ndarray
    psi_ey_xhi: jnp.ndarray
    psi_ey_zlo: jnp.ndarray
    psi_ey_zhi: jnp.ndarray
    psi_ez_xlo: jnp.ndarray
    psi_ez_xhi: jnp.ndarray
    psi_ez_ylo: jnp.ndarray
    psi_ez_yhi: jnp.ndarray
    # Psi arrays for H-field update (from E-curl correction)
    psi_hx_ylo: jnp.ndarray
    psi_hx_yhi: jnp.ndarray
    psi_hx_zlo: jnp.ndarray
    psi_hx_zhi: jnp.ndarray
    psi_hy_xlo: jnp.ndarray
    psi_hy_xhi: jnp.ndarray
    psi_hy_zlo: jnp.ndarray
    psi_hy_zhi: jnp.ndarray
    psi_hz_xlo: jnp.ndarray
    psi_hz_xhi: jnp.ndarray
    psi_hz_ylo: jnp.ndarray
    psi_hz_yhi: jnp.ndarray




def _cpml_profile(
    n_layers: int,
    dt,
    dx,
    order: int = 3,
    kappa_max: float = 1.0,
    R_asymptotic: float = 1e-15,
) -> CPMLParams:
    """Compute graded CPML profile using polynomial grading.

    Parameters
    ----------
    n_layers : int
        Number of CPML layers.
    dt : float
        Timestep (seconds).
    dx : float
        Cell size (meters).
    order : int
        Polynomial grading order.  Default 2 (quadratic, matching Meep).
    kappa_max : float
        Maximum κ stretching parameter. 1.0 = standard CPML (no stretching).
        Values > 1 improve evanescent wave absorption.
    R_asymptotic : float
        Target asymptotic reflection coefficient.  σ_max is derived from
        this value using the Meep formula:
        ``σ_max = -ln(R) * (m+1) / (2 * η * d)``
        where ``d = n_layers * dx``.  Default 1e-15 (matching Meep).
    """
    EPS_0 = 8.854187817e-12
    MU_0 = 1.2566370614e-6

    # Stay in-trace when dt/dx are JAX tracers (mesh-as-design-variable
    # path); fall back to numpy for the standard concrete-grid case so
    # CPMLParams stays a constant pytree under JIT.  The concrete path
    # keeps its original float64 intermediate computation (cast to
    # float32 only at the CPMLParams boundary) so the returned profile
    # is bit-identical to the pre-#45 release.  The traced path stays
    # at float32 to match the default JAX dtype policy.
    traced = is_tracer(dt) or is_tracer(dx)
    xp = jnp if traced else np
    work_dtype = jnp.float32 if traced else np.float64

    eta = float(np.sqrt(MU_0 / EPS_0))  # ≈ 376.73 Ω (constant)
    d = n_layers * dx             # PML physical thickness
    # σ_max from target reflection (Meep formula):
    #   R = exp(-2 * σ_max * d / ((m+1) * η))
    #   => σ_max = -ln(R) * (m+1) / (2 * η * d)
    sigma_max = -float(np.log(R_asymptotic)) * (order + 1) / (2.0 * eta * d)
    # For CFS-CPML (κ>1), scale σ_max by κ_max (Gedney recommendation).
    sigma_max = sigma_max * kappa_max

    # Graded profiles: polynomial from max at outer boundary (index 0)
    # to 0 at interior edge (index n-1) for the lo face.
    # The hi face uses jnp.flip() to reverse this.
    rho = 1.0 - xp.arange(n_layers, dtype=work_dtype) / max(n_layers - 1, 1)
    sigma = sigma_max * rho**order
    # κ graded from kappa_max (outer) to 1.0 (inner): κ(ρ) = 1 + (κ_max - 1) * ρ^m
    kappa = 1.0 + (kappa_max - 1.0) * rho**order
    alpha = 0.05 * (1.0 - rho)  # α: small, decreasing toward outer boundary

    # Update coefficients
    denom = sigma * kappa + kappa**2 * alpha
    b = xp.exp(-(sigma / kappa + alpha) * dt / EPS_0)
    c = xp.where(denom > 1e-30, sigma * (b - 1.0) / denom, 0.0)

    return CPMLParams(
        sigma=jnp.asarray(sigma, dtype=jnp.float32),
        kappa=jnp.asarray(kappa, dtype=jnp.float32),
        alpha=jnp.asarray(alpha, dtype=jnp.float32),
        b=jnp.asarray(b, dtype=jnp.float32),
        c=jnp.asarray(c, dtype=jnp.float32),
    )


def _get_axis_cell_sizes(grid):
    """Extract per-axis cell sizes from Grid or NonUniformGrid.

    Returns (dx, dy, dz_lo, dz_hi) where dz_lo/dz_hi are the constant
    cell sizes in the z-lo and z-hi CPML padding regions.  For uniform
    grids all four values equal grid.dx.

    On the mesh-as-design-variable path, ``dz_arr`` may be a JAX tracer;
    indexed reads are preserved in-trace (no ``float()`` cast).
    """
    dx = float(grid.dx)
    dy = float(getattr(grid, 'dy', dx))
    dz_arr = getattr(grid, 'dz', None)
    if dz_arr is not None and len(dz_arr) > 0:
        if is_tracer(dz_arr):
            dz_lo = dz_arr[0]
            dz_hi = dz_arr[-1]
        else:
            dz_lo = float(dz_arr[0])
            dz_hi = float(dz_arr[-1])
    else:
        dz_lo = dx
        dz_hi = dx
    return dx, dy, dz_lo, dz_hi


def _pad_profile_at_end(p: CPMLParams, n_active: int, n_alloc: int) -> CPMLParams:
    """Extend a lo-face profile from ``n_active`` to ``n_alloc`` by
    appending no-op entries (σ=0, κ=1, α=0, b=1, c=0) at the end.

    Used by T7 Phase 2 PR2 when the user requests a lo-face CPML
    thickness below the allocation budget. The padded region (indices
    ``[n_active:n_alloc]``) produces identity updates — no absorption,
    no field modification — so the cells between the active CPML and
    the simulation interior behave like plain Yee cells.
    """
    if n_active >= n_alloc:
        return p
    pad = n_alloc - n_active
    return CPMLParams(
        sigma=jnp.concatenate([p.sigma, jnp.zeros(pad, dtype=p.sigma.dtype)]),
        kappa=jnp.concatenate([p.kappa, jnp.ones(pad, dtype=p.kappa.dtype)]),
        alpha=jnp.concatenate([p.alpha, jnp.zeros(pad, dtype=p.alpha.dtype)]),
        b=jnp.concatenate([p.b, jnp.ones(pad, dtype=p.b.dtype)]),
        c=jnp.concatenate([p.c, jnp.zeros(pad, dtype=p.c.dtype)]),
    )


def _pad_profile_at_start(p: CPMLParams, n_active: int, n_alloc: int) -> CPMLParams:
    """Prepend no-op entries to bring a hi-face (pre-flipped) profile
    up to ``n_alloc``. The active cells stay at the outer boundary end
    of the array (index ``n_alloc - 1``), and the front padding
    corresponds to interior cells that see no absorption.
    """
    if n_active >= n_alloc:
        return p
    pad = n_alloc - n_active
    return CPMLParams(
        sigma=jnp.concatenate([jnp.zeros(pad, dtype=p.sigma.dtype), p.sigma]),
        kappa=jnp.concatenate([jnp.ones(pad, dtype=p.kappa.dtype), p.kappa]),
        alpha=jnp.concatenate([jnp.zeros(pad, dtype=p.alpha.dtype), p.alpha]),
        b=jnp.concatenate([jnp.ones(pad, dtype=p.b.dtype), p.b]),
        c=jnp.concatenate([jnp.zeros(pad, dtype=p.c.dtype), p.c]),
    )


def _flip_profile(p: CPMLParams) -> CPMLParams:
    """Return the hi-face profile for a lo-oriented CPMLParams.

    Pre-computes the flipped (σ, κ, α, b, c) arrays so the scan body
    can access hi-face coefficients via direct field read instead of
    calling ``jnp.flip`` at every step (T7 Phase 2 PR1).
    """
    return CPMLParams(
        sigma=jnp.flip(p.sigma),
        kappa=jnp.flip(p.kappa),
        alpha=jnp.flip(p.alpha),
        b=jnp.flip(p.b),
        c=jnp.flip(p.c),
    )


def _cpml_noop_profile(n_layers: int) -> CPMLParams:
    """Return a no-op CPML profile (identity: no absorption).

    Used for faces where PEC boundary is desired instead of CPML.
    With b=1, c=0, kappa=1, the CPML update becomes a no-op and
    the natural PEC boundary condition takes effect.
    """
    return CPMLParams(
        sigma=jnp.zeros(n_layers, dtype=jnp.float32),
        kappa=jnp.ones(n_layers, dtype=jnp.float32),
        alpha=jnp.zeros(n_layers, dtype=jnp.float32),
        b=jnp.ones(n_layers, dtype=jnp.float32),
        c=jnp.zeros(n_layers, dtype=jnp.float32),
    )


def init_cpml(grid, *, kappa_max: float | None = None,
              pec_faces: set[str] | None = None,
              pmc_faces: set[str] | None = None) -> tuple[CPMLAxisParams, CPMLState]:
    """Initialize CPML parameters and zero-valued auxiliary fields.

    Creates per-axis CPML profiles so that each face uses the correct
    cell size for its sigma_max and curl scaling.  On uniform grids
    (dx=dy=dz) all profiles are identical (backward compatible).

    Parameters
    ----------
    grid : Grid or NonUniformGrid (duck-typed)
        Simulation grid.  Reads ``dx``, optionally ``dy`` and ``dz``.
    kappa_max : float or None
        Maximum κ stretching parameter for CFS-CPML.
    pec_faces : set of str or None
        Faces to force PEC (no absorption).  Valid names:
        ``"x_lo"``, ``"x_hi"``, ``"y_lo"``, ``"y_hi"``,
        ``"z_lo"``, ``"z_hi"``.  Default: None (CPML on all faces).
    pmc_faces : set of str or None
        Faces carrying a PMC reflector (v1.7.5 per-face padding puts
        ``pad=0`` cells on that side of the axis). Like ``pec_faces``
        these faces receive a no-op CPML profile — otherwise the
        ``apply_cpml_e/h`` slices ``[:, :n, :]`` etc. would apply
        absorber coefficients to the first ``n`` *interior* cells on
        the PMC side and dissipate energy that should reflect.
    """
    if kappa_max is None:
        kappa_max = getattr(grid, "kappa_max", None) or 1.0
    if pec_faces is None:
        pec_faces = getattr(grid, "pec_faces", None) or set()
    if pmc_faces is None:
        pmc_faces = getattr(grid, "pmc_faces", None) or set()
    # PMC and PEC are electromagnetic duals — both are perfect
    # conductors — and both need the CPML profile suppressed so the
    # absorber does not eat the first n interior cells on that side.
    noop_faces = set(pec_faces) | set(pmc_faces)
    n = grid.cpml_layers
    dx, dy, dz_lo, dz_hi = _get_axis_cell_sizes(grid)

    noop = _cpml_noop_profile(n)
    # T7 Phase 2 PR2: per-face active layer counts. The allocation
    # budget ``n`` stays uniform (CPMLState shape unchanged, JIT cache
    # preserved on the symmetric common case). Faces with an active
    # layer count below the budget get a no-op-padded profile so the
    # scan body sees identity updates in the padded region.
    face_layers = getattr(grid, "face_layers", None) or {
        f: n for f in ("x_lo", "x_hi", "y_lo", "y_hi", "z_lo", "z_hi")
    }

    def _lo_face_profile(is_noop: bool, cell_size, face_name: str) -> CPMLParams:
        if is_noop:
            return noop
        n_active = int(face_layers.get(face_name, n))
        p = _cpml_profile(n_active, grid.dt, cell_size, kappa_max=kappa_max)
        return _pad_profile_at_end(p, n_active, n)

    def _hi_face_profile(is_noop: bool, cell_size, face_name: str) -> CPMLParams:
        if is_noop:
            return noop
        n_active = int(face_layers.get(face_name, n))
        base = _cpml_profile(n_active, grid.dt, cell_size, kappa_max=kappa_max)
        return _pad_profile_at_start(_flip_profile(base), n_active, n)

    prof_x_lo = _lo_face_profile("x_lo" in noop_faces, dx, "x_lo")
    prof_x_hi = _hi_face_profile("x_hi" in noop_faces, dx, "x_hi")
    prof_y_lo = _lo_face_profile("y_lo" in noop_faces, dy, "y_lo")
    prof_y_hi = _hi_face_profile("y_hi" in noop_faces, dy, "y_hi")
    prof_z_lo = _lo_face_profile("z_lo" in noop_faces, dz_lo, "z_lo")
    prof_z_hi = _hi_face_profile("z_hi" in noop_faces, dz_hi, "z_hi")

    params = CPMLAxisParams(
        x_lo=prof_x_lo, x_hi=prof_x_hi,
        y_lo=prof_y_lo, y_hi=prof_y_hi,
        z_lo=prof_z_lo, z_hi=prof_z_hi,
        dx_x_lo=dx, dx_x_hi=dx,
        dx_y_lo=dy, dx_y_hi=dy,
        dz_lo=dz_lo, dz_hi=dz_hi,
    )

    nx, ny, nz = grid.shape if hasattr(grid, 'shape') else (grid.nx, grid.ny, grid.nz)

    def _zeros(dim_size: int, perp1: int, perp2: int) -> jnp.ndarray:
        return jnp.zeros((n, perp1, perp2), dtype=jnp.float32)

    state = CPMLState(
        # E-field psi (12 faces)
        psi_ex_ylo=_zeros(n, nx, nz), psi_ex_yhi=_zeros(n, nx, nz),
        psi_ex_zlo=_zeros(n, nx, ny), psi_ex_zhi=_zeros(n, nx, ny),
        psi_ey_xlo=_zeros(n, ny, nz), psi_ey_xhi=_zeros(n, ny, nz),
        psi_ey_zlo=_zeros(n, ny, nx), psi_ey_zhi=_zeros(n, ny, nx),
        psi_ez_xlo=_zeros(n, nz, ny), psi_ez_xhi=_zeros(n, nz, ny),
        psi_ez_ylo=_zeros(n, nz, nx), psi_ez_yhi=_zeros(n, nz, nx),
        # H-field psi (12 faces)
        psi_hx_ylo=_zeros(n, nx, nz), psi_hx_yhi=_zeros(n, nx, nz),
        psi_hx_zlo=_zeros(n, nx, ny), psi_hx_zhi=_zeros(n, nx, ny),
        psi_hy_xlo=_zeros(n, ny, nz), psi_hy_xhi=_zeros(n, ny, nz),
        psi_hy_zlo=_zeros(n, ny, nx), psi_hy_zhi=_zeros(n, ny, nx),
        psi_hz_xlo=_zeros(n, nz, ny), psi_hz_xhi=_zeros(n, nz, ny),
        psi_hz_ylo=_zeros(n, nz, nx), psi_hz_yhi=_zeros(n, nz, nx),
    )

    return params, state


def _kappa_correction(kappa, curl_slice, shape_broadcast):
    """Compute the κ correction term: (1/κ - 1) * curl.

    When κ=1, this returns 0 (no correction). When κ>1, this reduces
    the curl contribution inside the PML region by dividing by κ.

    Parameters
    ----------
    kappa : array (n_layers,)
        κ profile values.
    curl_slice : array
        The curl component slice in the PML region.
    shape_broadcast : str
        How to reshape kappa for broadcasting. One of 'x', 'y', 'z'.
    """
    # Reshape kappa for broadcasting to 3D slice
    k = kappa[:, None, None]  # (n, 1, 1)
    return (1.0 / k - 1.0) * curl_slice


def apply_cpml_e(
    state, cpml_params, cpml_state: CPMLState, grid,
    axes: str = "xyz", materials=None,  # materials reserved for future per-cell CPML
) -> tuple:
    """Apply CPML correction to E-field update on all 6 faces.

    Accepts either legacy ``CPMLParams`` (single profile, backward
    compatible) or ``CPMLAxisParams`` (per-axis profiles for
    anisotropic grids).
    """
    n = grid.cpml_layers
    dt = grid.dt
    EPS_0 = 8.854187817e-12
    # Material-aware CPML: use local eps_r so guided modes in dielectric
    # waveguides see an impedance-matched absorber (equivalent to UPML).
    # Falls back to free-space eps_0 when materials is None.
    if materials is not None:
        _ce_full = dt / (materials.eps_r * EPS_0)  # (nx, ny, nz)
        # Pre-slice to each PML face region for broadcasting with psi arrays
        ce_xlo = _ce_full[:n, :, :];  ce_xhi = _ce_full[-n:, :, :]
        ce_ylo = _ce_full[:, :n, :];  ce_yhi = _ce_full[:, -n:, :]
        ce_zlo = _ce_full[:, :, :n];  ce_zhi = _ce_full[:, :, -n:]
    else:
        ce_xlo = ce_xhi = ce_ylo = ce_yhi = ce_zlo = ce_zhi = dt / EPS_0

    # Unpack per-face profiles and cell sizes (T7 Phase 2 PR1 — 6 faces).
    if isinstance(cpml_params, CPMLAxisParams):
        px_lo, px_hi = cpml_params.x_lo, cpml_params.x_hi
        py_lo, py_hi = cpml_params.y_lo, cpml_params.y_hi
        pz_lo, pz_hi = cpml_params.z_lo, cpml_params.z_hi
        dx_x_lo, dx_x_hi = cpml_params.dx_x_lo, cpml_params.dx_x_hi
        dx_y_lo, dx_y_hi = cpml_params.dx_y_lo, cpml_params.dx_y_hi
        dz_lo, dz_hi = cpml_params.dz_lo, cpml_params.dz_hi
    else:
        # Legacy single-profile path (uniform grid): build ephemeral
        # flipped hi-face profiles so the scan body stays uniform.
        px_lo = py_lo = pz_lo = cpml_params
        px_hi = py_hi = pz_hi = _flip_profile(cpml_params)
        dx_x_lo = dx_x_hi = dx_y_lo = dx_y_hi = dz_lo = dz_hi = float(grid.dx)

    # X-axis profiles (hi-face pre-flipped at init time — no jnp.flip here).
    b_x_lo = px_lo.b[:, None, None]; c_x_lo = px_lo.c[:, None, None]; k_x_lo = px_lo.kappa[:, None, None]
    b_x_hi = px_hi.b[:, None, None]; c_x_hi = px_hi.c[:, None, None]; k_x_hi = px_hi.kappa[:, None, None]
    # Y-axis profiles
    b_y_lo = py_lo.b[:, None, None]; c_y_lo = py_lo.c[:, None, None]; k_y_lo = py_lo.kappa[:, None, None]
    b_y_hi = py_hi.b[:, None, None]; c_y_hi = py_hi.c[:, None, None]; k_y_hi = py_hi.kappa[:, None, None]
    # Z-axis profiles (lo and hi independent by design)
    b_zl = pz_lo.b[:, None, None]; c_zl = pz_lo.c[:, None, None]; k_zl = pz_lo.kappa[:, None, None]
    b_zh = pz_hi.b[:, None, None]; c_zh = pz_hi.c[:, None, None]; k_zh = pz_hi.kappa[:, None, None]

    ex = state.ex
    ey = state.ey
    ez = state.ez

    # =========================================================
    # X-axis CPML: Ey correction (dHz/dx) and Ez correction (dHy/dx)
    # =========================================================

    if "x" in axes:
        # --- X-lo: Ey correction from dHz/dx ---
        hz_xlo = state.hz[:n, :, :]
        hz_shifted_xlo = _shift_bwd(state.hz, 0)[:n, :, :]
        curl_hz_dx_xlo = (hz_xlo - hz_shifted_xlo) / dx_x_lo

        new_psi_ey_xlo = b_x_lo * cpml_state.psi_ey_xlo + c_x_lo * curl_hz_dx_xlo
        ey = ey.at[:n, :, :].add(-ce_xlo * new_psi_ey_xlo)
        ey = ey.at[:n, :, :].add(-ce_xlo * (1.0 / k_x_lo - 1.0) * curl_hz_dx_xlo)

        # --- X-hi: Ey correction from dHz/dx ---
        hz_xhi = state.hz[-n:, :, :]
        hz_shifted_xhi = _shift_bwd(state.hz, 0)[-n:, :, :]
        curl_hz_dx_xhi = (hz_xhi - hz_shifted_xhi) / dx_x_hi

        new_psi_ey_xhi = b_x_hi * cpml_state.psi_ey_xhi + c_x_hi * curl_hz_dx_xhi
        ey = ey.at[-n:, :, :].add(-ce_xhi * new_psi_ey_xhi)
        ey = ey.at[-n:, :, :].add(-ce_xhi * (1.0 / k_x_hi - 1.0) * curl_hz_dx_xhi)

        # --- X-lo: Ez correction from dHy/dx ---
        hy_xlo = state.hy[:n, :, :]
        hy_shifted_xlo = _shift_bwd(state.hy, 0)[:n, :, :]
        curl_hy_dx_xlo = (hy_xlo - hy_shifted_xlo) / dx_x_lo
        curl_hy_dx_xlo_t = jnp.transpose(curl_hy_dx_xlo, (0, 2, 1))

        new_psi_ez_xlo = b_x_lo * cpml_state.psi_ez_xlo + c_x_lo * curl_hy_dx_xlo_t
        correction_ez_xlo = jnp.transpose(new_psi_ez_xlo, (0, 2, 1))
        ez = ez.at[:n, :, :].add(ce_xlo * correction_ez_xlo)
        ez = ez.at[:n, :, :].add(ce_xlo * (1.0 / k_x_lo - 1.0) * curl_hy_dx_xlo)

        # --- X-hi: Ez correction from dHy/dx ---
        hy_xhi = state.hy[-n:, :, :]
        hy_shifted_xhi = _shift_bwd(state.hy, 0)[-n:, :, :]
        curl_hy_dx_xhi = (hy_xhi - hy_shifted_xhi) / dx_x_hi
        curl_hy_dx_xhi_t = jnp.transpose(curl_hy_dx_xhi, (0, 2, 1))

        new_psi_ez_xhi = b_x_hi * cpml_state.psi_ez_xhi + c_x_hi * curl_hy_dx_xhi_t
        correction_ez_xhi = jnp.transpose(new_psi_ez_xhi, (0, 2, 1))
        ez = ez.at[-n:, :, :].add(ce_xhi * correction_ez_xhi)
        ez = ez.at[-n:, :, :].add(ce_xhi * (1.0 / k_x_hi - 1.0) * curl_hy_dx_xhi)
    else:
        new_psi_ey_xlo = cpml_state.psi_ey_xlo
        new_psi_ey_xhi = cpml_state.psi_ey_xhi
        new_psi_ez_xlo = cpml_state.psi_ez_xlo
        new_psi_ez_xhi = cpml_state.psi_ez_xhi

    # =========================================================
    # Y-axis CPML: Ex correction (dHz/dy) and Ez correction (dHx/dy)
    # =========================================================

    if "y" in axes:
        # --- Y-lo: Ex correction from dHz/dy ---
        hz_ylo = state.hz[:, :n, :]
        hz_shifted_ylo = _shift_bwd(state.hz, 1)[:, :n, :]
        curl_hz_dy_ylo = (hz_ylo - hz_shifted_ylo) / dx_y_lo

        curl_hz_dy_ylo_t = jnp.transpose(curl_hz_dy_ylo, (1, 0, 2))

        new_psi_ex_ylo = b_y_lo * cpml_state.psi_ex_ylo + c_y_lo * curl_hz_dy_ylo_t
        correction_ex_ylo = jnp.transpose(new_psi_ex_ylo, (1, 0, 2))
        ex = ex.at[:, :n, :].add(ce_ylo * correction_ex_ylo)
        kappa_corr_ylo = jnp.transpose((1.0 / k_y_lo - 1.0) * curl_hz_dy_ylo_t, (1, 0, 2))
        ex = ex.at[:, :n, :].add(ce_ylo * kappa_corr_ylo)

        # --- Y-hi: Ex correction from dHz/dy ---
        hz_yhi = state.hz[:, -n:, :]
        hz_shifted_yhi = _shift_bwd(state.hz, 1)[:, -n:, :]
        curl_hz_dy_yhi = (hz_yhi - hz_shifted_yhi) / dx_y_hi

        curl_hz_dy_yhi_t = jnp.transpose(curl_hz_dy_yhi, (1, 0, 2))

        new_psi_ex_yhi = b_y_hi * cpml_state.psi_ex_yhi + c_y_hi * curl_hz_dy_yhi_t
        correction_ex_yhi = jnp.transpose(new_psi_ex_yhi, (1, 0, 2))
        ex = ex.at[:, -n:, :].add(ce_yhi * correction_ex_yhi)
        kappa_corr_yhi = jnp.transpose((1.0 / k_y_hi - 1.0) * curl_hz_dy_yhi_t, (1, 0, 2))
        ex = ex.at[:, -n:, :].add(ce_yhi * kappa_corr_yhi)

        # --- Y-lo: Ez correction from dHx/dy ---
        hx_ylo = state.hx[:, :n, :]
        hx_shifted_ylo = _shift_bwd(state.hx, 1)[:, :n, :]
        curl_hx_dy_ylo = (hx_ylo - hx_shifted_ylo) / dx_y_lo

        curl_hx_dy_ylo_t = jnp.transpose(curl_hx_dy_ylo, (1, 2, 0))

        new_psi_ez_ylo = b_y_lo * cpml_state.psi_ez_ylo + c_y_lo * curl_hx_dy_ylo_t
        correction_ez_ylo = jnp.transpose(new_psi_ez_ylo, (2, 0, 1))
        ez = ez.at[:, :n, :].add(-ce_ylo * correction_ez_ylo)
        kappa_corr_ez_ylo = jnp.transpose((1.0 / k_y_lo - 1.0) * curl_hx_dy_ylo_t, (2, 0, 1))
        ez = ez.at[:, :n, :].add(-ce_ylo * kappa_corr_ez_ylo)

        # --- Y-hi: Ez correction from dHx/dy ---
        hx_yhi = state.hx[:, -n:, :]
        hx_shifted_yhi = _shift_bwd(state.hx, 1)[:, -n:, :]
        curl_hx_dy_yhi = (hx_yhi - hx_shifted_yhi) / dx_y_hi

        curl_hx_dy_yhi_t = jnp.transpose(curl_hx_dy_yhi, (1, 2, 0))

        new_psi_ez_yhi = b_y_hi * cpml_state.psi_ez_yhi + c_y_hi * curl_hx_dy_yhi_t
        correction_ez_yhi = jnp.transpose(new_psi_ez_yhi, (2, 0, 1))
        ez = ez.at[:, -n:, :].add(-ce_yhi * correction_ez_yhi)
        kappa_corr_ez_yhi = jnp.transpose((1.0 / k_y_hi - 1.0) * curl_hx_dy_yhi_t, (2, 0, 1))
        ez = ez.at[:, -n:, :].add(-ce_yhi * kappa_corr_ez_yhi)
    else:
        new_psi_ex_ylo = cpml_state.psi_ex_ylo
        new_psi_ex_yhi = cpml_state.psi_ex_yhi
        new_psi_ez_ylo = cpml_state.psi_ez_ylo
        new_psi_ez_yhi = cpml_state.psi_ez_yhi

    # =========================================================
    # Z-axis CPML: Ex correction (dHy/dz) and Ey correction (dHx/dz)
    # =========================================================

    if "z" in axes:
        # --- Z-lo: Ex correction from dHy/dz ---
        hy_zlo = state.hy[:, :, :n]
        hy_shifted_zlo = _shift_bwd(state.hy, 2)[:, :, :n]
        curl_hy_dz_zlo = (hy_zlo - hy_shifted_zlo) / dz_lo

        curl_hy_dz_zlo_t = jnp.transpose(curl_hy_dz_zlo, (2, 0, 1))

        new_psi_ex_zlo = b_zl * cpml_state.psi_ex_zlo + c_zl * curl_hy_dz_zlo_t
        correction_ex_zlo = jnp.transpose(new_psi_ex_zlo, (1, 2, 0))
        ex = ex.at[:, :, :n].add(-ce_zlo * correction_ex_zlo)
        # κ correction for Z-lo Ex
        kappa_corr_ex_zlo = jnp.transpose((1.0 / k_zl - 1.0) * curl_hy_dz_zlo_t, (1, 2, 0))
        ex = ex.at[:, :, :n].add(-ce_zlo * kappa_corr_ex_zlo)

        # --- Z-hi: Ex correction from dHy/dz ---
        hy_zhi = state.hy[:, :, -n:]
        hy_shifted_zhi = _shift_bwd(state.hy, 2)[:, :, -n:]
        curl_hy_dz_zhi = (hy_zhi - hy_shifted_zhi) / dz_hi

        curl_hy_dz_zhi_t = jnp.transpose(curl_hy_dz_zhi, (2, 0, 1))

        new_psi_ex_zhi = b_zh * cpml_state.psi_ex_zhi + c_zh * curl_hy_dz_zhi_t
        correction_ex_zhi = jnp.transpose(new_psi_ex_zhi, (1, 2, 0))
        ex = ex.at[:, :, -n:].add(-ce_zhi * correction_ex_zhi)
        kappa_corr_ex_zhi = jnp.transpose((1.0 / k_zh - 1.0) * curl_hy_dz_zhi_t, (1, 2, 0))
        ex = ex.at[:, :, -n:].add(-ce_zhi * kappa_corr_ex_zhi)

        # --- Z-lo: Ey correction from dHx/dz ---
        hx_zlo = state.hx[:, :, :n]
        hx_shifted_zlo = _shift_bwd(state.hx, 2)[:, :, :n]
        curl_hx_dz_zlo = (hx_zlo - hx_shifted_zlo) / dz_lo

        curl_hx_dz_zlo_t = jnp.transpose(curl_hx_dz_zlo, (2, 1, 0))

        new_psi_ey_zlo = b_zl * cpml_state.psi_ey_zlo + c_zl * curl_hx_dz_zlo_t
        correction_ey_zlo = jnp.transpose(new_psi_ey_zlo, (2, 1, 0))
        ey = ey.at[:, :, :n].add(ce_zlo * correction_ey_zlo)
        kappa_corr_ey_zlo = jnp.transpose((1.0 / k_zl - 1.0) * curl_hx_dz_zlo_t, (2, 1, 0))
        ey = ey.at[:, :, :n].add(ce_zlo * kappa_corr_ey_zlo)

        # --- Z-hi: Ey correction from dHx/dz ---
        hx_zhi = state.hx[:, :, -n:]
        hx_shifted_zhi = _shift_bwd(state.hx, 2)[:, :, -n:]
        curl_hx_dz_zhi = (hx_zhi - hx_shifted_zhi) / dz_hi

        curl_hx_dz_zhi_t = jnp.transpose(curl_hx_dz_zhi, (2, 1, 0))

        new_psi_ey_zhi = b_zh * cpml_state.psi_ey_zhi + c_zh * curl_hx_dz_zhi_t
        correction_ey_zhi = jnp.transpose(new_psi_ey_zhi, (2, 1, 0))
        ey = ey.at[:, :, -n:].add(ce_zhi * correction_ey_zhi)
        kappa_corr_ey_zhi = jnp.transpose((1.0 / k_zh - 1.0) * curl_hx_dz_zhi_t, (2, 1, 0))
        ey = ey.at[:, :, -n:].add(ce_zhi * kappa_corr_ey_zhi)
    else:
        new_psi_ex_zlo = cpml_state.psi_ex_zlo
        new_psi_ex_zhi = cpml_state.psi_ex_zhi
        new_psi_ey_zlo = cpml_state.psi_ey_zlo
        new_psi_ey_zhi = cpml_state.psi_ey_zhi

    state = state._replace(ex=ex, ey=ey, ez=ez)
    cpml_state = cpml_state._replace(
        # x-axis
        psi_ey_xlo=new_psi_ey_xlo,
        psi_ey_xhi=new_psi_ey_xhi,
        psi_ez_xlo=new_psi_ez_xlo,
        psi_ez_xhi=new_psi_ez_xhi,
        # y-axis
        psi_ex_ylo=new_psi_ex_ylo,
        psi_ex_yhi=new_psi_ex_yhi,
        psi_ez_ylo=new_psi_ez_ylo,
        psi_ez_yhi=new_psi_ez_yhi,
        # z-axis
        psi_ex_zlo=new_psi_ex_zlo,
        psi_ex_zhi=new_psi_ex_zhi,
        psi_ey_zlo=new_psi_ey_zlo,
        psi_ey_zhi=new_psi_ey_zhi,
    )

    return state, cpml_state


def apply_cpml_h(
    state, cpml_params, cpml_state: CPMLState, grid,
    axes: str = "xyz", materials=None,
) -> tuple:
    """Apply CPML correction to H-field update on all 6 faces.

    Accepts either legacy ``CPMLParams`` or ``CPMLAxisParams``.
    When *materials* is provided, uses local mu_r for impedance-matched
    guided-mode absorption (equivalent to UPML).
    """
    n = grid.cpml_layers
    dt = grid.dt
    MU_0 = 1.2566370614e-6
    if materials is not None and hasattr(materials, 'mu_r'):
        _ch_full = dt / (materials.mu_r * MU_0)  # (nx, ny, nz)
        ch_xlo = _ch_full[:n, :, :];  ch_xhi = _ch_full[-n:, :, :]
        ch_ylo = _ch_full[:, :n, :];  ch_yhi = _ch_full[:, -n:, :]
        ch_zlo = _ch_full[:, :, :n];  ch_zhi = _ch_full[:, :, -n:]
    else:
        ch_xlo = ch_xhi = ch_ylo = ch_yhi = ch_zlo = ch_zhi = dt / MU_0

    # Unpack per-face profiles and cell sizes (T7 Phase 2 PR1).
    if isinstance(cpml_params, CPMLAxisParams):
        px_lo, px_hi = cpml_params.x_lo, cpml_params.x_hi
        py_lo, py_hi = cpml_params.y_lo, cpml_params.y_hi
        pz_lo, pz_hi = cpml_params.z_lo, cpml_params.z_hi
        dx_x_lo, dx_x_hi = cpml_params.dx_x_lo, cpml_params.dx_x_hi
        dx_y_lo, dx_y_hi = cpml_params.dx_y_lo, cpml_params.dx_y_hi
        dz_lo, dz_hi = cpml_params.dz_lo, cpml_params.dz_hi
    else:
        px_lo = py_lo = pz_lo = cpml_params
        px_hi = py_hi = pz_hi = _flip_profile(cpml_params)
        dx_x_lo = dx_x_hi = dx_y_lo = dx_y_hi = dz_lo = dz_hi = float(grid.dx)

    # X-axis profiles (hi-face pre-flipped at init time).
    b_x_lo = px_lo.b[:, None, None]; c_x_lo = px_lo.c[:, None, None]; k_x_lo = px_lo.kappa[:, None, None]
    b_x_hi = px_hi.b[:, None, None]; c_x_hi = px_hi.c[:, None, None]; k_x_hi = px_hi.kappa[:, None, None]
    # Y-axis profiles
    b_y_lo = py_lo.b[:, None, None]; c_y_lo = py_lo.c[:, None, None]; k_y_lo = py_lo.kappa[:, None, None]
    b_y_hi = py_hi.b[:, None, None]; c_y_hi = py_hi.c[:, None, None]; k_y_hi = py_hi.kappa[:, None, None]
    # Z-axis profiles
    b_zl = pz_lo.b[:, None, None]; c_zl = pz_lo.c[:, None, None]; k_zl = pz_lo.kappa[:, None, None]
    b_zh = pz_hi.b[:, None, None]; c_zh = pz_hi.c[:, None, None]; k_zh = pz_hi.kappa[:, None, None]

    hx = state.hx
    hy = state.hy
    hz = state.hz

    # =========================================================
    # X-axis CPML: Hy correction (dEz/dx) and Hz correction (dEy/dx)
    # =========================================================

    if "x" in axes:
        # --- X-lo: Hy correction from dEz/dx ---
        ez_xlo = state.ez[:n, :, :]
        ez_shifted_xlo = _shift_fwd(state.ez, 0)[:n, :, :]
        curl_ez_dx_xlo = (ez_shifted_xlo - ez_xlo) / dx_x_lo

        new_psi_hy_xlo = b_x_lo * cpml_state.psi_hy_xlo + c_x_lo * curl_ez_dx_xlo
        hy = hy.at[:n, :, :].add(ch_xlo * new_psi_hy_xlo)
        hy = hy.at[:n, :, :].add(ch_xlo * (1.0 / k_x_lo - 1.0) * curl_ez_dx_xlo)

        # --- X-hi: Hy correction from dEz/dx ---
        ez_xhi = state.ez[-n:, :, :]
        ez_shifted_xhi = _shift_fwd(state.ez, 0)[-n:, :, :]
        curl_ez_dx_xhi = (ez_shifted_xhi - ez_xhi) / dx_x_hi

        new_psi_hy_xhi = b_x_hi * cpml_state.psi_hy_xhi + c_x_hi * curl_ez_dx_xhi
        hy = hy.at[-n:, :, :].add(ch_xhi * new_psi_hy_xhi)
        hy = hy.at[-n:, :, :].add(ch_xhi * (1.0 / k_x_hi - 1.0) * curl_ez_dx_xhi)

        # --- X-lo: Hz correction from dEy/dx ---
        ey_xlo = state.ey[:n, :, :]
        ey_shifted_xlo = _shift_fwd(state.ey, 0)[:n, :, :]
        curl_ey_dx_xlo = (ey_shifted_xlo - ey_xlo) / dx_x_lo
        curl_ey_dx_xlo_t = jnp.transpose(curl_ey_dx_xlo, (0, 2, 1))

        new_psi_hz_xlo = b_x_lo * cpml_state.psi_hz_xlo + c_x_lo * curl_ey_dx_xlo_t
        correction_hz_xlo = jnp.transpose(new_psi_hz_xlo, (0, 2, 1))
        hz = hz.at[:n, :, :].add(-ch_xlo * correction_hz_xlo)
        hz = hz.at[:n, :, :].add(-ch_xlo * (1.0 / k_x_lo - 1.0) * curl_ey_dx_xlo)

        # --- X-hi: Hz correction from dEy/dx ---
        ey_xhi = state.ey[-n:, :, :]
        ey_shifted_xhi = _shift_fwd(state.ey, 0)[-n:, :, :]
        curl_ey_dx_xhi = (ey_shifted_xhi - ey_xhi) / dx_x_hi
        curl_ey_dx_xhi_t = jnp.transpose(curl_ey_dx_xhi, (0, 2, 1))

        new_psi_hz_xhi = b_x_hi * cpml_state.psi_hz_xhi + c_x_hi * curl_ey_dx_xhi_t
        correction_hz_xhi = jnp.transpose(new_psi_hz_xhi, (0, 2, 1))
        hz = hz.at[-n:, :, :].add(-ch_xhi * correction_hz_xhi)
        hz = hz.at[-n:, :, :].add(-ch_xhi * (1.0 / k_x_hi - 1.0) * curl_ey_dx_xhi)
    else:
        new_psi_hy_xlo = cpml_state.psi_hy_xlo
        new_psi_hy_xhi = cpml_state.psi_hy_xhi
        new_psi_hz_xlo = cpml_state.psi_hz_xlo
        new_psi_hz_xhi = cpml_state.psi_hz_xhi

    # =========================================================
    # Y-axis CPML: Hx correction (dEz/dy) and Hz correction (dEx/dy)
    # =========================================================

    if "y" in axes:
        # --- Y-lo: Hx correction from dEz/dy ---
        ez_ylo = state.ez[:, :n, :]
        ez_shifted_ylo = _shift_fwd(state.ez, 1)[:, :n, :]
        curl_ez_dy_ylo = (ez_shifted_ylo - ez_ylo) / dx_y_lo

        curl_ez_dy_ylo_t = jnp.transpose(curl_ez_dy_ylo, (1, 0, 2))

        new_psi_hx_ylo = b_y_lo * cpml_state.psi_hx_ylo + c_y_lo * curl_ez_dy_ylo_t
        correction_hx_ylo = jnp.transpose(new_psi_hx_ylo, (1, 0, 2))
        hx = hx.at[:, :n, :].add(-ch_ylo * correction_hx_ylo)
        kappa_corr_hx_ylo = jnp.transpose((1.0 / k_y_lo - 1.0) * curl_ez_dy_ylo_t, (1, 0, 2))
        hx = hx.at[:, :n, :].add(-ch_ylo * kappa_corr_hx_ylo)

        # --- Y-hi: Hx correction from dEz/dy ---
        ez_yhi = state.ez[:, -n:, :]
        ez_shifted_yhi = _shift_fwd(state.ez, 1)[:, -n:, :]
        curl_ez_dy_yhi = (ez_shifted_yhi - ez_yhi) / dx_y_hi

        curl_ez_dy_yhi_t = jnp.transpose(curl_ez_dy_yhi, (1, 0, 2))

        new_psi_hx_yhi = b_y_hi * cpml_state.psi_hx_yhi + c_y_hi * curl_ez_dy_yhi_t
        correction_hx_yhi = jnp.transpose(new_psi_hx_yhi, (1, 0, 2))
        hx = hx.at[:, -n:, :].add(-ch_yhi * correction_hx_yhi)
        kappa_corr_hx_yhi = jnp.transpose((1.0 / k_y_hi - 1.0) * curl_ez_dy_yhi_t, (1, 0, 2))
        hx = hx.at[:, -n:, :].add(-ch_yhi * kappa_corr_hx_yhi)

        # --- Y-lo: Hz correction from dEx/dy ---
        ex_ylo = state.ex[:, :n, :]
        ex_shifted_ylo = _shift_fwd(state.ex, 1)[:, :n, :]
        curl_ex_dy_ylo = (ex_shifted_ylo - ex_ylo) / dx_y_lo

        curl_ex_dy_ylo_t = jnp.transpose(curl_ex_dy_ylo, (1, 2, 0))

        new_psi_hz_ylo = b_y_lo * cpml_state.psi_hz_ylo + c_y_lo * curl_ex_dy_ylo_t
        correction_hz_ylo = jnp.transpose(new_psi_hz_ylo, (2, 0, 1))
        hz = hz.at[:, :n, :].add(ch_ylo * correction_hz_ylo)
        kappa_corr_hz_ylo = jnp.transpose((1.0 / k_y_lo - 1.0) * curl_ex_dy_ylo_t, (2, 0, 1))
        hz = hz.at[:, :n, :].add(ch_ylo * kappa_corr_hz_ylo)

        # --- Y-hi: Hz correction from dEx/dy ---
        ex_yhi = state.ex[:, -n:, :]
        ex_shifted_yhi = _shift_fwd(state.ex, 1)[:, -n:, :]
        curl_ex_dy_yhi = (ex_shifted_yhi - ex_yhi) / dx_y_hi

        curl_ex_dy_yhi_t = jnp.transpose(curl_ex_dy_yhi, (1, 2, 0))

        new_psi_hz_yhi = b_y_hi * cpml_state.psi_hz_yhi + c_y_hi * curl_ex_dy_yhi_t
        correction_hz_yhi = jnp.transpose(new_psi_hz_yhi, (2, 0, 1))
        hz = hz.at[:, -n:, :].add(ch_yhi * correction_hz_yhi)
        kappa_corr_hz_yhi = jnp.transpose((1.0 / k_y_hi - 1.0) * curl_ex_dy_yhi_t, (2, 0, 1))
        hz = hz.at[:, -n:, :].add(ch_yhi * kappa_corr_hz_yhi)
    else:
        new_psi_hx_ylo = cpml_state.psi_hx_ylo
        new_psi_hx_yhi = cpml_state.psi_hx_yhi
        new_psi_hz_ylo = cpml_state.psi_hz_ylo
        new_psi_hz_yhi = cpml_state.psi_hz_yhi

    # =========================================================
    # Z-axis CPML: Hx correction (dEy/dz) and Hy correction (dEx/dz)
    # =========================================================

    if "z" in axes:
        # --- Z-lo: Hx correction from dEy/dz ---
        ey_zlo = state.ey[:, :, :n]
        ey_shifted_zlo = _shift_fwd(state.ey, 2)[:, :, :n]
        curl_ey_dz_zlo = (ey_shifted_zlo - ey_zlo) / dz_lo

        curl_ey_dz_zlo_t = jnp.transpose(curl_ey_dz_zlo, (2, 0, 1))

        new_psi_hx_zlo = b_zl * cpml_state.psi_hx_zlo + c_zl * curl_ey_dz_zlo_t
        correction_hx_zlo = jnp.transpose(new_psi_hx_zlo, (1, 2, 0))
        hx = hx.at[:, :, :n].add(ch_zlo * correction_hx_zlo)
        kappa_corr_hx_zlo = jnp.transpose((1.0 / k_zl - 1.0) * curl_ey_dz_zlo_t, (1, 2, 0))
        hx = hx.at[:, :, :n].add(ch_zlo * kappa_corr_hx_zlo)

        # --- Z-hi: Hx correction from dEy/dz ---
        ey_zhi = state.ey[:, :, -n:]
        ey_shifted_zhi = _shift_fwd(state.ey, 2)[:, :, -n:]
        curl_ey_dz_zhi = (ey_shifted_zhi - ey_zhi) / dz_hi

        curl_ey_dz_zhi_t = jnp.transpose(curl_ey_dz_zhi, (2, 0, 1))

        new_psi_hx_zhi = b_zh * cpml_state.psi_hx_zhi + c_zh * curl_ey_dz_zhi_t
        correction_hx_zhi = jnp.transpose(new_psi_hx_zhi, (1, 2, 0))
        hx = hx.at[:, :, -n:].add(ch_zhi * correction_hx_zhi)
        kappa_corr_hx_zhi = jnp.transpose((1.0 / k_zh - 1.0) * curl_ey_dz_zhi_t, (1, 2, 0))
        hx = hx.at[:, :, -n:].add(ch_zhi * kappa_corr_hx_zhi)

        # --- Z-lo: Hy correction from dEx/dz ---
        ex_zlo = state.ex[:, :, :n]
        ex_shifted_zlo = _shift_fwd(state.ex, 2)[:, :, :n]
        curl_ex_dz_zlo = (ex_shifted_zlo - ex_zlo) / dz_lo

        curl_ex_dz_zlo_t = jnp.transpose(curl_ex_dz_zlo, (2, 1, 0))

        new_psi_hy_zlo = b_zl * cpml_state.psi_hy_zlo + c_zl * curl_ex_dz_zlo_t
        correction_hy_zlo = jnp.transpose(new_psi_hy_zlo, (2, 1, 0))
        hy = hy.at[:, :, :n].add(-ch_zlo * correction_hy_zlo)
        kappa_corr_hy_zlo = jnp.transpose((1.0 / k_zl - 1.0) * curl_ex_dz_zlo_t, (2, 1, 0))
        hy = hy.at[:, :, :n].add(-ch_zlo * kappa_corr_hy_zlo)

        # --- Z-hi: Hy correction from dEx/dz ---
        ex_zhi = state.ex[:, :, -n:]
        ex_shifted_zhi = _shift_fwd(state.ex, 2)[:, :, -n:]
        curl_ex_dz_zhi = (ex_shifted_zhi - ex_zhi) / dz_hi

        curl_ex_dz_zhi_t = jnp.transpose(curl_ex_dz_zhi, (2, 1, 0))

        new_psi_hy_zhi = b_zh * cpml_state.psi_hy_zhi + c_zh * curl_ex_dz_zhi_t
        correction_hy_zhi = jnp.transpose(new_psi_hy_zhi, (2, 1, 0))
        hy = hy.at[:, :, -n:].add(-ch_zhi * correction_hy_zhi)
        kappa_corr_hy_zhi = jnp.transpose((1.0 / k_zh - 1.0) * curl_ex_dz_zhi_t, (2, 1, 0))
        hy = hy.at[:, :, -n:].add(-ch_zhi * kappa_corr_hy_zhi)
    else:
        new_psi_hx_zlo = cpml_state.psi_hx_zlo
        new_psi_hx_zhi = cpml_state.psi_hx_zhi
        new_psi_hy_zlo = cpml_state.psi_hy_zlo
        new_psi_hy_zhi = cpml_state.psi_hy_zhi

    state = state._replace(hx=hx, hy=hy, hz=hz)
    cpml_state = cpml_state._replace(
        # x-axis
        psi_hy_xlo=new_psi_hy_xlo,
        psi_hy_xhi=new_psi_hy_xhi,
        psi_hz_xlo=new_psi_hz_xlo,
        psi_hz_xhi=new_psi_hz_xhi,
        # y-axis
        psi_hx_ylo=new_psi_hx_ylo,
        psi_hx_yhi=new_psi_hx_yhi,
        psi_hz_ylo=new_psi_hz_ylo,
        psi_hz_yhi=new_psi_hz_yhi,
        # z-axis
        psi_hx_zlo=new_psi_hx_zlo,
        psi_hx_zhi=new_psi_hx_zhi,
        psi_hy_zlo=new_psi_hy_zlo,
        psi_hy_zhi=new_psi_hy_zhi,
    )

    return state, cpml_state
