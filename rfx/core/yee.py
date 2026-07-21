"""Yee cell FDTD update equations — pure JAX functions.

All functions are jax.jit-compatible and operate on explicit state arrays.
No hidden mutable state.
"""

from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp


class FDTDState(NamedTuple):
    """Complete FDTD simulation state at one timestep."""

    # Electric field components (Nx, Ny, Nz)
    ex: jnp.ndarray
    ey: jnp.ndarray
    ez: jnp.ndarray
    # Magnetic field components (Nx, Ny, Nz)
    hx: jnp.ndarray
    hy: jnp.ndarray
    hz: jnp.ndarray
    # Timestep counter
    step: jnp.ndarray


class MaterialArrays(NamedTuple):
    """Material property arrays on the grid."""

    # Relative permittivity (Nx, Ny, Nz) — used in E update
    eps_r: jnp.ndarray
    # Conductivity S/m (Nx, Ny, Nz) — used in E update (lossy)
    sigma: jnp.ndarray
    # Relative permeability (Nx, Ny, Nz) — used in H update
    mu_r: jnp.ndarray


def init_state(shape: tuple[int, int, int], *, field_dtype=jnp.float32) -> FDTDState:
    """Initialize zero-valued FDTD state.

    Parameters
    ----------
    shape : (Nx, Ny, Nz)
    field_dtype : jnp dtype
        Data type for field arrays.  Use ``jnp.float16`` for mixed-
        precision mode (material coefficients and accumulators stay
        float32; only field storage is reduced).
    """
    zeros = jnp.zeros(shape, dtype=field_dtype)
    return FDTDState(
        ex=zeros, ey=zeros, ez=zeros,
        hx=zeros, hy=zeros, hz=zeros,
        step=jnp.array(0, dtype=jnp.int32),
    )


def init_materials(shape: tuple[int, int, int]) -> MaterialArrays:
    """Initialize free-space material arrays."""
    ones = jnp.ones(shape, dtype=jnp.float32)
    return MaterialArrays(
        eps_r=ones,
        sigma=jnp.zeros(shape, dtype=jnp.float32),
        mu_r=ones,
    )


# Physical constants — post-2019 SI / CODATA. EPS_0 and MU_0 form a
# mutually consistent pair: 1/sqrt(MU_0 * EPS_0) == c (299792458 m/s),
# so the FDTD's effective speed of light is exact. Changing one without
# the other breaks that consistency (see the TE10-cutoff gate).
EPS_0 = 8.8541878128e-12  # F/m — post-2019 SI / CODATA value
MU_0 = 1.25663706212e-6  # H/m — post-2019 SI / CODATA value


def _shift_fwd(arr, axis):
    """arr[i+1] with zero at the last position (replaces roll(arr, -1, axis))."""
    pad_widths = [(0, 0)] * arr.ndim
    pad_widths[axis] = (0, 1)
    padded = jnp.pad(arr, pad_widths)
    slices = [slice(None)] * arr.ndim
    slices[axis] = slice(1, None)
    return padded[tuple(slices)]


def _shift_bwd(arr, axis):
    """arr[i-1] with zero at the first position (replaces roll(arr, +1, axis))."""
    pad_widths = [(0, 0)] * arr.ndim
    pad_widths[axis] = (1, 0)
    padded = jnp.pad(arr, pad_widths)
    slices = [slice(None)] * arr.ndim
    slices[axis] = slice(None, -1)
    return padded[tuple(slices)]


# --- (2,4) fourth-order-in-space staggered stencil (Fang 1996 / standard) ---
# A 4th-order option for the smooth-bulk lever (analytic gate-1: ~2.52x fewer
# cells/wavelength than (2,2) for the same dispersion error; gate-2 spike:
# this advantage is a SMOOTH-PROPAGATION property and does NOT extend to PEC /
# geometric features, which staircase at 2nd order regardless — see
# docs/research_notes/20260627_memory_efficiency_techniques_exploration.md).
# The two coefficients on the near (1/2-cell) and far (3/2-cell) staggered
# differences. At a non-periodic boundary the wider stencil reaches outside the
# domain, so the first/last TWO gaps revert to 2nd order (the 2-cell PEC/edge
# ribbon — a 1-cell ribbon leaves the i=N-2 fwd / i=1 bwd gap malformed).
_C4_NEAR = 9.0 / 8.0
_C4_FAR = -1.0 / 24.0


def _ribbon_2nd(d4, d2, axis):
    """Revert the first TWO and last TWO slices along ``axis`` to the 2nd-order
    difference. The (2,4) far term f[i+2]-f[i-1] (fwd) / f[i+1]-f[i-2] (bwd)
    reaches outside the domain for any gap within 2 cells of a non-periodic face;
    those zero-padded far terms form a malformed stencil, so revert them. Width-2
    covers every boundary-affected gap for BOTH directions (over-reverts at most
    one interior-valid cell — negligible in the large domains 4th order targets;
    for tiny axes it safely degenerates to all-2nd-order)."""
    lead = [slice(None)] * d4.ndim
    lead[axis] = slice(0, 2)
    trail = [slice(None)] * d4.ndim
    trail[axis] = slice(-2, None)
    d4 = d4.at[tuple(lead)].set(d2[tuple(lead)])
    d4 = d4.at[tuple(trail)].set(d2[tuple(trail)])
    return d4


def _diff_fwd_o(arr, axis, periodic, order, bloch=None):
    """Forward staggered first difference (f[i+1]-f[i] family), at i+1/2; the
    caller divides by dx. order=2 is byte-identical to ``_shift_fwd(arr)-arr``
    when ``bloch is None``.

    ``bloch`` (oblique-periodic Bloch field-transformation, #404): a length-3
    tuple of per-axis complex phases ``exp(-j·k_axis·dx)``.  On a PERIODIC axis
    the forward-rolled neighbour is multiplied by ``bloch[axis]`` so the plain
    ``jnp.roll`` wrap represents the exact discrete Yee derivative of a wave with
    transverse wavenumber ``k_axis``.  ``bloch=None`` (default) leaves the real
    path untouched; ``bloch`` is only threaded on order-2 (guarded upstream)."""
    if order == 2:
        if periodic[axis]:
            nxt = jnp.roll(arr, -1, axis)
            if bloch is not None:
                nxt = nxt * bloch[axis]
            return nxt - arr
        return _shift_fwd(arr, axis) - arr
    # order == 4:  c1*(f[i+1]-f[i]) + c2*(f[i+2]-f[i-1])
    if periodic[axis]:
        near = jnp.roll(arr, -1, axis) - arr
        far = jnp.roll(arr, -2, axis) - jnp.roll(arr, 1, axis)
        return _C4_NEAR * near + _C4_FAR * far
    near = _shift_fwd(arr, axis) - arr
    far = _shift_fwd(_shift_fwd(arr, axis), axis) - _shift_bwd(arr, axis)
    return _ribbon_2nd(_C4_NEAR * near + _C4_FAR * far, near, axis)


def _diff_bwd_o(arr, axis, periodic, order, bloch=None):
    """Backward staggered first difference (f[i]-f[i-1] family), at i-1/2; the
    caller divides by dx. order=2 is byte-identical to ``arr-_shift_bwd(arr)``
    when ``bloch is None``.

    For the Bloch field-transformation (#404) the BACKWARD-rolled neighbour
    carries the CONJUGATE phase ``exp(+j·k_axis·dx)`` (``bloch[axis].conjugate()``),
    the exact discrete adjoint of the forward stagger.  See ``_diff_fwd_o``."""
    if order == 2:
        if periodic[axis]:
            prv = jnp.roll(arr, 1, axis)
            if bloch is not None:
                prv = prv * bloch[axis].conjugate()
            return arr - prv
        return arr - _shift_bwd(arr, axis)
    # order == 4:  c1*(f[i]-f[i-1]) + c2*(f[i+1]-f[i-2])
    if periodic[axis]:
        near = arr - jnp.roll(arr, 1, axis)
        far = jnp.roll(arr, -1, axis) - jnp.roll(arr, 2, axis)
        return _C4_NEAR * near + _C4_FAR * far
    near = arr - _shift_bwd(arr, axis)
    far = _shift_fwd(arr, axis) - _shift_bwd(_shift_bwd(arr, axis), axis)
    return _ribbon_2nd(_C4_NEAR * near + _C4_FAR * far, near, axis)


@partial(jax.jit, static_argnums=(4, 5, 6))
def update_h(state: FDTDState, materials: MaterialArrays, dt: float, dx: float,
             periodic: tuple = (False, False, False),
             stencil_order: int = 2,
             bloch: tuple | None = None) -> FDTDState:
    """Magnetic field half-step update (Faraday's law).

    H^{n+1/2} = H^{n-1/2} - (dt / μ) * curl(E^n)

    Note: uses a single ``dx`` for all three axes (cubic cells: dx=dy=dz).
    For non-uniform z grids, use ``update_h_nu`` / ``update_e_nu`` instead.

    periodic: tuple of 3 bools selecting periodic boundary per axis (x, y, z).
    stencil_order: 2 (default, byte-identical) or 4 (Fang (2,4) wide stencil,
        reverting to 2nd order in the 1-cell ribbon at non-periodic faces).
    bloch: None (default, real path, byte-identical) or a length-3 tuple of
        per-axis complex phases ``exp(-j·k_axis·dx)`` for the oblique-periodic
        Bloch field-transformation (#404).  When set, the fields must be a
        complex dtype (the transformed envelope P); requires stencil_order=2.
    """
    so = stencil_order
    if so not in (2, 4):
        raise ValueError(f"stencil_order must be 2 or 4, got {so}")
    if bloch is not None and so != 2:
        raise ValueError(
            f"bloch phase (oblique-periodic BC, #404) requires stencil_order=2, got {so}"
        )

    # Compute dtype: the oblique-periodic Bloch path carries a complex field
    # envelope P; the real path stays float32 (upcast for reduced-precision
    # fields) and is byte-identical.
    _fdtype = state.ex.dtype
    _cdtype = jnp.complex64 if jnp.iscomplexobj(state.ex) else jnp.float32
    ex = state.ex.astype(_cdtype)
    ey = state.ey.astype(_cdtype)
    ez = state.ez.astype(_cdtype)
    mu = materials.mu_r * MU_0

    # curl E components via forward staggered differences (order=2 byte-identical)
    # dEz/dy - dEy/dz
    curl_x = (
        _diff_fwd_o(ez, 1, periodic, so, bloch) / dx
        - _diff_fwd_o(ey, 2, periodic, so, bloch) / dx
    )
    # dEx/dz - dEz/dx
    curl_y = (
        _diff_fwd_o(ex, 2, periodic, so, bloch) / dx
        - _diff_fwd_o(ez, 0, periodic, so, bloch) / dx
    )
    # dEy/dx - dEx/dy
    curl_z = (
        _diff_fwd_o(ey, 0, periodic, so, bloch) / dx
        - _diff_fwd_o(ex, 1, periodic, so, bloch) / dx
    )

    hx = (state.hx.astype(_cdtype) - (dt / mu) * curl_x).astype(_fdtype)
    hy = (state.hy.astype(_cdtype) - (dt / mu) * curl_y).astype(_fdtype)
    hz = (state.hz.astype(_cdtype) - (dt / mu) * curl_z).astype(_fdtype)

    return state._replace(hx=hx, hy=hy, hz=hz)


@partial(jax.jit, static_argnums=(4, 5, 6))
def update_e(state: FDTDState, materials: MaterialArrays, dt: float, dx: float,
             periodic: tuple = (False, False, False),
             stencil_order: int = 2,
             bloch: tuple | None = None) -> FDTDState:
    """Electric field full-step update (Ampere's law).

    For lossy media with conductivity σ:
    E^{n+1} = Ca * E^n + Cb * curl(H^{n+1/2})

    Ca = (1 - σ*dt/(2ε)) / (1 + σ*dt/(2ε))
    Cb = (dt/ε) / (1 + σ*dt/(2ε))

    periodic: tuple of 3 bools selecting periodic boundary per axis (x, y, z).
    stencil_order: 2 (default, byte-identical) or 4 (Fang (2,4) wide stencil,
        reverting to 2nd order in the 1-cell ribbon at non-periodic faces).
    bloch: None (default, real path, byte-identical) or a length-3 tuple of
        per-axis complex phases ``exp(-j·k_axis·dx)`` for the oblique-periodic
        Bloch field-transformation (#404); requires stencil_order=2 and complex
        fields.  The backward differences use the conjugate phase automatically.
    """
    so = stencil_order
    if so not in (2, 4):
        raise ValueError(f"stencil_order must be 2 or 4, got {so}")
    if bloch is not None and so != 2:
        raise ValueError(
            f"bloch phase (oblique-periodic BC, #404) requires stencil_order=2, got {so}"
        )

    # Compute dtype: complex for the Bloch envelope path, float32 for the real
    # path (byte-identical; upcast covers reduced-precision fields).
    _fdtype = state.ex.dtype
    _cdtype = jnp.complex64 if jnp.iscomplexobj(state.ex) else jnp.float32
    hx = state.hx.astype(_cdtype)
    hy = state.hy.astype(_cdtype)
    hz = state.hz.astype(_cdtype)
    eps = materials.eps_r * EPS_0
    sigma = materials.sigma

    # Lossy update coefficients
    sigma_dt_2eps = sigma * dt / (2.0 * eps)
    ca = (1.0 - sigma_dt_2eps) / (1.0 + sigma_dt_2eps)
    cb = (dt / eps) / (1.0 + sigma_dt_2eps)

    # curl H components via backward staggered differences (order=2 byte-identical)
    # dHz/dy - dHy/dz
    curl_x = (
        _diff_bwd_o(hz, 1, periodic, so, bloch) / dx
        - _diff_bwd_o(hy, 2, periodic, so, bloch) / dx
    )
    # dHx/dz - dHz/dx
    curl_y = (
        _diff_bwd_o(hx, 2, periodic, so, bloch) / dx
        - _diff_bwd_o(hz, 0, periodic, so, bloch) / dx
    )
    # dHy/dx - dHx/dy
    curl_z = (
        _diff_bwd_o(hy, 0, periodic, so, bloch) / dx
        - _diff_bwd_o(hx, 1, periodic, so, bloch) / dx
    )

    ex = (ca * state.ex.astype(_cdtype) + cb * curl_x).astype(_fdtype)
    ey = (ca * state.ey.astype(_cdtype) + cb * curl_y).astype(_fdtype)
    ez = (ca * state.ez.astype(_cdtype) + cb * curl_z).astype(_fdtype)

    return state._replace(
        ex=ex, ey=ey, ez=ez,
        step=state.step + 1,
    )


# ---------------------------------------------------------------------------
# Non-uniform mesh updates
# ---------------------------------------------------------------------------

def update_h_nu(state: FDTDState, materials: MaterialArrays, dt: float,
                inv_dx_h: jnp.ndarray, inv_dy_h: jnp.ndarray, inv_dz_h: jnp.ndarray,
                ) -> FDTDState:
    """H update for non-uniform Yee grid.

    The H-curl differences two E nodes that straddle a single cell, so
    it divides by the LOCAL cell width (CORE-C2 fix, 2026-05-16).

    Parameters
    ----------
    inv_dx_h, inv_dy_h, inv_dz_h : (N,) arrays
        H-update inverse spacing ``inv_d_h[k] = 1/d[k]`` for k<N-1,
        ``inv_d_h[N-1] = 0``. Built by
        ``rfx.nonuniform._profile_to_inv_arrays`` (second return value).
    """
    _fdtype = state.ex.dtype
    ex = state.ex.astype(jnp.float32)
    ey = state.ey.astype(jnp.float32)
    ez = state.ez.astype(jnp.float32)
    mu = materials.mu_r * MU_0

    # Forward differences with same shape (zero-pad via _shift_fwd)
    curl_x = (
        (_shift_fwd(ez, 1) - ez) * inv_dy_h[None, :, None]
        - (_shift_fwd(ey, 2) - ey) * inv_dz_h[None, None, :]
    )
    curl_y = (
        (_shift_fwd(ex, 2) - ex) * inv_dz_h[None, None, :]
        - (_shift_fwd(ez, 0) - ez) * inv_dx_h[:, None, None]
    )
    curl_z = (
        (_shift_fwd(ey, 0) - ey) * inv_dx_h[:, None, None]
        - (_shift_fwd(ex, 1) - ex) * inv_dy_h[None, :, None]
    )

    hx = (state.hx.astype(jnp.float32) - (dt / mu) * curl_x).astype(_fdtype)
    hy = (state.hy.astype(jnp.float32) - (dt / mu) * curl_y).astype(_fdtype)
    hz = (state.hz.astype(jnp.float32) - (dt / mu) * curl_z).astype(_fdtype)

    return state._replace(hx=hx, hy=hy, hz=hz)


def update_e_nu(state: FDTDState, materials: MaterialArrays, dt: float,
                inv_dx: jnp.ndarray, inv_dy: jnp.ndarray, inv_dz: jnp.ndarray,
                ) -> FDTDState:
    """E update for non-uniform Yee grid.

    The E-curl differences two H cell-centres, whose separation is the
    MEAN of the two adjacent cell widths (CORE-C2 fix, 2026-05-16).

    Parameters
    ----------
    inv_dx, inv_dy, inv_dz : (N,) arrays
        E-update inverse spacing ``inv_d_e[k] = 2/(d[k-1]+d[k])`` for
        k>=1, ``inv_d_e[0] = 1/d[0]``. Built by
        ``rfx.nonuniform._profile_to_inv_arrays`` (first return value).
    """
    _fdtype = state.ex.dtype
    hx = state.hx.astype(jnp.float32)
    hy = state.hy.astype(jnp.float32)
    hz = state.hz.astype(jnp.float32)
    eps = materials.eps_r * EPS_0
    sigma = materials.sigma

    sigma_dt_2eps = sigma * dt / (2.0 * eps)
    ca = (1.0 - sigma_dt_2eps) / (1.0 + sigma_dt_2eps)
    cb = (dt / eps) / (1.0 + sigma_dt_2eps)

    # Backward differences with same shape (zero-pad via _shift_bwd)
    curl_x = (
        (hz - _shift_bwd(hz, 1)) * inv_dy[None, :, None]
        - (hy - _shift_bwd(hy, 2)) * inv_dz[None, None, :]
    )
    curl_y = (
        (hx - _shift_bwd(hx, 2)) * inv_dz[None, None, :]
        - (hz - _shift_bwd(hz, 0)) * inv_dx[:, None, None]
    )
    curl_z = (
        (hy - _shift_bwd(hy, 0)) * inv_dx[:, None, None]
        - (hx - _shift_bwd(hx, 1)) * inv_dy[None, :, None]
    )

    ex = (ca * state.ex.astype(jnp.float32) + cb * curl_x).astype(_fdtype)
    ey = (ca * state.ey.astype(jnp.float32) + cb * curl_y).astype(_fdtype)
    ez = (ca * state.ez.astype(jnp.float32) + cb * curl_z).astype(_fdtype)

    return state._replace(ex=ex, ey=ey, ez=ez, step=state.step + 1)


# ---------------------------------------------------------------------------
# Pre-computed update coefficients for high-throughput scan loops
# ---------------------------------------------------------------------------

class UpdateCoeffs(NamedTuple):
    """Pre-computed FDTD update coefficients.

    Eliminates per-step coefficient recomputation and optionally bakes in
    PEC boundary enforcement (zero coefficients at boundary cells) to
    remove the need for separate ``apply_pec()`` calls.

    Use :func:`precompute_coeffs` to build.
    """
    # H-field coefficient: dt / (mu_r * MU_0 * dx)  — shape (Nx, Ny, Nz)
    ch: jnp.ndarray
    # E-field decay coefficient  — per-component (Nx, Ny, Nz)
    ca_ex: jnp.ndarray
    ca_ey: jnp.ndarray
    ca_ez: jnp.ndarray
    # E-field curl coefficient (includes 1/dx)  — per-component (Nx, Ny, Nz)
    cb_ex: jnp.ndarray
    cb_ey: jnp.ndarray
    cb_ez: jnp.ndarray


def precompute_coeffs(
    materials: MaterialArrays,
    dt: float,
    dx: float,
    *,
    pec_axes: str = "",
) -> UpdateCoeffs:
    """Pre-compute all FDTD update coefficients.

    Parameters
    ----------
    materials : MaterialArrays
    dt, dx : float
    pec_axes : str
        Axes on which to bake PEC (zero tangential E) into the
        coefficients.  For example ``"xyz"`` zeros Ca/Cb at all 6
        boundary faces so that ``apply_pec()`` is no longer needed.

    Returns
    -------
    UpdateCoeffs
    """
    ch = jnp.float32(dt / (MU_0 * dx)) / materials.mu_r

    eps = materials.eps_r * jnp.float32(EPS_0)
    sigma = materials.sigma
    loss = sigma * jnp.float32(dt) / (jnp.float32(2.0) * eps)
    denom = jnp.float32(1.0) + loss
    ca = (jnp.float32(1.0) - loss) / denom
    cb_over_dx = (jnp.float32(dt) / eps) / (denom * jnp.float32(dx))

    # Start with isotropic coefficients.
    ca_ex = ca
    ca_ey = ca
    ca_ez = ca
    cb_ex = cb_over_dx
    cb_ey = cb_over_dx
    cb_ez = cb_over_dx

    # Bake PEC boundary enforcement into the coefficients by zeroing
    # Ca and Cb at boundary faces for tangential E components.
    if pec_axes:
        nx, ny, nz = materials.eps_r.shape
        if "x" in pec_axes:
            # Ey, Ez tangential on x-faces
            ca_ey = ca_ey.at[0, :, :].set(0.0).at[-1, :, :].set(0.0)
            ca_ez = ca_ez.at[0, :, :].set(0.0).at[-1, :, :].set(0.0)
            cb_ey = cb_ey.at[0, :, :].set(0.0).at[-1, :, :].set(0.0)
            cb_ez = cb_ez.at[0, :, :].set(0.0).at[-1, :, :].set(0.0)
        if "y" in pec_axes:
            # Ex, Ez tangential on y-faces
            ca_ex = ca_ex.at[:, 0, :].set(0.0).at[:, -1, :].set(0.0)
            ca_ez = ca_ez.at[:, 0, :].set(0.0).at[:, -1, :].set(0.0)
            cb_ex = cb_ex.at[:, 0, :].set(0.0).at[:, -1, :].set(0.0)
            cb_ez = cb_ez.at[:, 0, :].set(0.0).at[:, -1, :].set(0.0)
        if "z" in pec_axes:
            # Ex, Ey tangential on z-faces
            ca_ex = ca_ex.at[:, :, 0].set(0.0).at[:, :, -1].set(0.0)
            ca_ey = ca_ey.at[:, :, 0].set(0.0).at[:, :, -1].set(0.0)
            cb_ex = cb_ex.at[:, :, 0].set(0.0).at[:, :, -1].set(0.0)
            cb_ey = cb_ey.at[:, :, 0].set(0.0).at[:, :, -1].set(0.0)

    return UpdateCoeffs(
        ch=ch,
        ca_ex=ca_ex, ca_ey=ca_ey, ca_ez=ca_ez,
        cb_ex=cb_ex, cb_ey=cb_ey, cb_ez=cb_ez,
    )


def update_h_fast(state: FDTDState, ch: jnp.ndarray) -> FDTDState:
    """H update using pre-computed coefficient ``ch = dt/(mu*dx)``.

    Avoids recomputing material coefficients each timestep.
    Non-periodic boundaries only (uses ``_shift_fwd``).
    """
    _fdtype = state.ex.dtype
    ex = state.ex.astype(jnp.float32)
    ey = state.ey.astype(jnp.float32)
    ez = state.ez.astype(jnp.float32)
    hx = (state.hx.astype(jnp.float32) - ch * ((_shift_fwd(ez, 1) - ez) - (_shift_fwd(ey, 2) - ey))).astype(_fdtype)
    hy = (state.hy.astype(jnp.float32) - ch * ((_shift_fwd(ex, 2) - ex) - (_shift_fwd(ez, 0) - ez))).astype(_fdtype)
    hz = (state.hz.astype(jnp.float32) - ch * ((_shift_fwd(ey, 0) - ey) - (_shift_fwd(ex, 1) - ex))).astype(_fdtype)
    return state._replace(hx=hx, hy=hy, hz=hz)


def update_e_fast(
    state: FDTDState,
    ca_ex: jnp.ndarray, ca_ey: jnp.ndarray, ca_ez: jnp.ndarray,
    cb_ex: jnp.ndarray, cb_ey: jnp.ndarray, cb_ez: jnp.ndarray,
) -> FDTDState:
    """E update using pre-computed per-component coefficients.

    ``ca_*`` and ``cb_*`` already include PEC zeroing when built with
    :func:`precompute_coeffs` and ``pec_axes``, so no separate
    ``apply_pec()`` call is needed.

    Non-periodic boundaries only (uses ``_shift_bwd``).
    """
    _fdtype = state.ex.dtype
    hx = state.hx.astype(jnp.float32)
    hy = state.hy.astype(jnp.float32)
    hz = state.hz.astype(jnp.float32)
    ex = (ca_ex * state.ex.astype(jnp.float32) + cb_ex * ((hz - _shift_bwd(hz, 1)) - (hy - _shift_bwd(hy, 2)))).astype(_fdtype)
    ey = (ca_ey * state.ey.astype(jnp.float32) + cb_ey * ((hx - _shift_bwd(hx, 2)) - (hz - _shift_bwd(hz, 0)))).astype(_fdtype)
    ez = (ca_ez * state.ez.astype(jnp.float32) + cb_ez * ((hy - _shift_bwd(hy, 0)) - (hx - _shift_bwd(hx, 1)))).astype(_fdtype)
    return state._replace(ex=ex, ey=ey, ez=ez, step=state.step + 1)


def update_he_fast(state: FDTDState, coeffs: UpdateCoeffs) -> FDTDState:
    """Combined H + E update using pre-computed :class:`UpdateCoeffs`.

    Performs a full leapfrog step (H half-step then E full-step) with
    PEC baked into the coefficients.  This is the fastest path for the
    common case of non-periodic boundaries with uniform mesh.
    """
    _fdtype = state.ex.dtype
    # --- H update (upcast to float32 for arithmetic) ---
    ex = state.ex.astype(jnp.float32)
    ey = state.ey.astype(jnp.float32)
    ez = state.ez.astype(jnp.float32)
    ch = coeffs.ch
    hx = (state.hx.astype(jnp.float32) - ch * ((_shift_fwd(ez, 1) - ez) - (_shift_fwd(ey, 2) - ey))).astype(_fdtype)
    hy = (state.hy.astype(jnp.float32) - ch * ((_shift_fwd(ex, 2) - ex) - (_shift_fwd(ez, 0) - ez))).astype(_fdtype)
    hz = (state.hz.astype(jnp.float32) - ch * ((_shift_fwd(ey, 0) - ey) - (_shift_fwd(ex, 1) - ex))).astype(_fdtype)
    # --- E update (with PEC baked into coefficients) ---
    # Upcast newly computed H fields back to float32 for curl computation
    hx_f = hx.astype(jnp.float32)
    hy_f = hy.astype(jnp.float32)
    hz_f = hz.astype(jnp.float32)
    ex = (coeffs.ca_ex * ex + coeffs.cb_ex * ((hz_f - _shift_bwd(hz_f, 1)) - (hy_f - _shift_bwd(hy_f, 2)))).astype(_fdtype)
    ey = (coeffs.ca_ey * ey + coeffs.cb_ey * ((hx_f - _shift_bwd(hx_f, 2)) - (hz_f - _shift_bwd(hz_f, 0)))).astype(_fdtype)
    ez = (coeffs.ca_ez * ez + coeffs.cb_ez * ((hy_f - _shift_bwd(hy_f, 0)) - (hx_f - _shift_bwd(hx_f, 1)))).astype(_fdtype)
    return FDTDState(ex=ex, ey=ey, ez=ez, hx=hx, hy=hy, hz=hz,
                     step=state.step + 1)


@jax.jit
def update_e_nu_aniso(state: FDTDState, materials: MaterialArrays,
                      eps_ex: jnp.ndarray, eps_ey: jnp.ndarray, eps_ez: jnp.ndarray,
                      dt: float,
                      inv_dx: jnp.ndarray, inv_dy: jnp.ndarray, inv_dz: jnp.ndarray,
                      ) -> FDTDState:
    """Non-uniform E update with per-component anisotropic permittivity.

    Same backward-difference structure as :func:`update_e_nu` (E-update
    mean spacing via ``inv_dx``/``inv_dy``/``inv_dz``) but uses three separate
    permittivity arrays (Ex, Ey, Ez) so subpixel-smoothing can be applied
    on a non-uniform mesh. ``materials.sigma`` is still applied
    isotropically (matches the uniform-path :func:`update_e_aniso`).
    """
    _fdtype = state.ex.dtype
    hx = state.hx.astype(jnp.float32)
    hy = state.hy.astype(jnp.float32)
    hz = state.hz.astype(jnp.float32)
    sigma = materials.sigma

    abs_eps_ex = eps_ex * EPS_0
    abs_eps_ey = eps_ey * EPS_0
    abs_eps_ez = eps_ez * EPS_0

    loss_ex = sigma * dt / (2.0 * abs_eps_ex)
    ca_ex = (1.0 - loss_ex) / (1.0 + loss_ex)
    cb_ex = (dt / abs_eps_ex) / (1.0 + loss_ex)

    loss_ey = sigma * dt / (2.0 * abs_eps_ey)
    ca_ey = (1.0 - loss_ey) / (1.0 + loss_ey)
    cb_ey = (dt / abs_eps_ey) / (1.0 + loss_ey)

    loss_ez = sigma * dt / (2.0 * abs_eps_ez)
    ca_ez = (1.0 - loss_ez) / (1.0 + loss_ez)
    cb_ez = (dt / abs_eps_ez) / (1.0 + loss_ez)

    # Backward differences with per-cell inv-spacing (mirrors update_e_nu)
    curl_x = (
        (hz - _shift_bwd(hz, 1)) * inv_dy[None, :, None]
        - (hy - _shift_bwd(hy, 2)) * inv_dz[None, None, :]
    )
    curl_y = (
        (hx - _shift_bwd(hx, 2)) * inv_dz[None, None, :]
        - (hz - _shift_bwd(hz, 0)) * inv_dx[:, None, None]
    )
    curl_z = (
        (hy - _shift_bwd(hy, 0)) * inv_dx[:, None, None]
        - (hx - _shift_bwd(hx, 1)) * inv_dy[None, :, None]
    )

    ex = (ca_ex * state.ex.astype(jnp.float32) + cb_ex * curl_x).astype(_fdtype)
    ey = (ca_ey * state.ey.astype(jnp.float32) + cb_ey * curl_y).astype(_fdtype)
    ez = (ca_ez * state.ez.astype(jnp.float32) + cb_ez * curl_z).astype(_fdtype)

    return state._replace(ex=ex, ey=ey, ez=ez, step=state.step + 1)


@partial(jax.jit, static_argnums=(7,))
def update_e_aniso_inv(state: FDTDState, materials: MaterialArrays,
                       inv_xx: jnp.ndarray, inv_yy: jnp.ndarray, inv_zz: jnp.ndarray,
                       dt: float, dx: float,
                       periodic: tuple = (False, False, False)) -> FDTDState:
    """Electric field update with per-component **inverse** permittivity.

    Stage 2 production form: takes the diagonal of the inverse-eps
    tensor (``inv_xx``, ``inv_yy``, ``inv_zz``) and uses it directly in
    the Yee update via *multiplication*. Numerically stable in the PEC
    limit (``inv = 0``): the Ca/Cb coefficients reduce to ``Ca = 1``,
    ``Cb = 0`` cleanly, freezing the field. Compare to
    :func:`update_e_aniso` (forward-eps form), which would compute
    ``1/(eps + 1e-30)`` and produce a huge but finite scaling — the
    NaN-trap path the original Stage 1 implementation hit.

    Derivation: ``stage2_ca_cb_derivation.md`` §5. Let ``μ = 1/ε_r``
    (per-component dimensionless inverse permittivity); then
    ``ε_abs = ε₀/μ``, so:

        loss = σ · dt · μ / (2 · ε₀)
        Ca   = (1 − loss) / (1 + loss)
        Cb   = (dt · μ / ε₀) / (1 + loss)
        E^{n+1} = Ca · E^n + Cb · curl(H^{n+1/2})

    For PEC tangential (``μ = 0``): loss = 0 → Ca = 1, Cb = 0 → field
    frozen. For dielectric (μ = 1/ε_r): same numerical form as the
    legacy ``update_e_aniso`` to within float-arithmetic ordering
    (~5 ULP). For partial-PEC perpendicular (``μ = (1−f)/ε_out``):
    finite scaling, no division hazard.

    Parameters
    ----------
    state : FDTDState
    materials : MaterialArrays
        Used only for ``sigma`` (conductivity, isotropic per cell).
        ``materials.eps_r`` is **not** consulted — the per-component
        inverse permittivity passed via ``inv_xx``/``inv_yy``/``inv_zz``
        is the source of truth.
    inv_xx, inv_yy, inv_zz : jnp.ndarray
        Per-component inverse-permittivity arrays (shape grid.shape,
        dtype float32). Typical sources:
        :func:`rfx.geometry.smoothing.compute_inv_eps_tensor_diag`.
    dt, dx : float
    periodic : tuple of 3 bools
    """
    def bwd(arr, axis):
        if periodic[axis]:
            return jnp.roll(arr, 1, axis)
        return _shift_bwd(arr, axis)

    _fdtype = state.ex.dtype
    hx = state.hx.astype(jnp.float32)
    hy = state.hy.astype(jnp.float32)
    hz = state.hz.astype(jnp.float32)
    sigma = materials.sigma

    # Per-component lossy update coefficients in inv-eps form.
    # `loss = σ · dt · μ / (2 · ε₀)` is finite for any (σ, μ) ≥ 0; the
    # `1 + loss` denominator is ≥ 1 so no division hazard.
    inv_eps0 = 1.0 / EPS_0
    loss_ex = 0.5 * sigma * dt * inv_xx * inv_eps0
    loss_ey = 0.5 * sigma * dt * inv_yy * inv_eps0
    loss_ez = 0.5 * sigma * dt * inv_zz * inv_eps0

    ca_ex = (1.0 - loss_ex) / (1.0 + loss_ex)
    ca_ey = (1.0 - loss_ey) / (1.0 + loss_ey)
    ca_ez = (1.0 - loss_ez) / (1.0 + loss_ez)

    cb_ex = (dt * inv_xx * inv_eps0) / (1.0 + loss_ex)
    cb_ey = (dt * inv_yy * inv_eps0) / (1.0 + loss_ey)
    cb_ez = (dt * inv_zz * inv_eps0) / (1.0 + loss_ez)

    # curl H (identical to update_e and update_e_aniso).
    curl_x = (
        (hz - bwd(hz, 1)) / dx
        - (hy - bwd(hy, 2)) / dx
    )
    curl_y = (
        (hx - bwd(hx, 2)) / dx
        - (hz - bwd(hz, 0)) / dx
    )
    curl_z = (
        (hy - bwd(hy, 0)) / dx
        - (hx - bwd(hx, 1)) / dx
    )

    ex = (ca_ex * state.ex.astype(jnp.float32) + cb_ex * curl_x).astype(_fdtype)
    ey = (ca_ey * state.ey.astype(jnp.float32) + cb_ey * curl_y).astype(_fdtype)
    ez = (ca_ez * state.ez.astype(jnp.float32) + cb_ez * curl_z).astype(_fdtype)

    return state._replace(
        ex=ex, ey=ey, ez=ez,
        step=state.step + 1,
    )


def update_e_aniso(state: FDTDState, materials: MaterialArrays,
                   eps_ex: jnp.ndarray, eps_ey: jnp.ndarray, eps_ez: jnp.ndarray,
                   dt: float, dx: float,
                   periodic: tuple = (False, False, False)) -> FDTDState:
    """Electric field update with per-component anisotropic permittivity.

    Same as :func:`update_e` but uses separate permittivity arrays for
    each E-field component (Ex, Ey, Ez) to support subpixel smoothing.

    The conductivity from ``materials.sigma`` is still applied isotropically.

    Parameters
    ----------
    state : FDTDState
    materials : MaterialArrays
        Used only for ``sigma`` (conductivity).
    eps_ex, eps_ey, eps_ez : jnp.ndarray
        Per-component relative permittivity arrays, each of shape (Nx, Ny, Nz).
    dt, dx : float
    periodic : tuple of 3 bools
    """
    def bwd(arr, axis):
        if periodic[axis]:
            return jnp.roll(arr, 1, axis)
        return _shift_bwd(arr, axis)

    _fdtype = state.ex.dtype
    hx = state.hx.astype(jnp.float32)
    hy = state.hy.astype(jnp.float32)
    hz = state.hz.astype(jnp.float32)
    sigma = materials.sigma

    # Per-component absolute permittivity
    abs_eps_ex = eps_ex * EPS_0
    abs_eps_ey = eps_ey * EPS_0
    abs_eps_ez = eps_ez * EPS_0

    # Per-component lossy update coefficients
    loss_ex = sigma * dt / (2.0 * abs_eps_ex)
    ca_ex = (1.0 - loss_ex) / (1.0 + loss_ex)
    cb_ex = (dt / abs_eps_ex) / (1.0 + loss_ex)

    loss_ey = sigma * dt / (2.0 * abs_eps_ey)
    ca_ey = (1.0 - loss_ey) / (1.0 + loss_ey)
    cb_ey = (dt / abs_eps_ey) / (1.0 + loss_ey)

    loss_ez = sigma * dt / (2.0 * abs_eps_ez)
    ca_ez = (1.0 - loss_ez) / (1.0 + loss_ez)
    cb_ez = (dt / abs_eps_ez) / (1.0 + loss_ez)

    # curl H (same as update_e)
    curl_x = (
        (hz - bwd(hz, 1)) / dx
        - (hy - bwd(hy, 2)) / dx
    )
    curl_y = (
        (hx - bwd(hx, 2)) / dx
        - (hz - bwd(hz, 0)) / dx
    )
    curl_z = (
        (hy - bwd(hy, 0)) / dx
        - (hx - bwd(hx, 1)) / dx
    )

    ex = (ca_ex * state.ex.astype(jnp.float32) + cb_ex * curl_x).astype(_fdtype)
    ey = (ca_ey * state.ey.astype(jnp.float32) + cb_ey * curl_y).astype(_fdtype)
    ez = (ca_ez * state.ez.astype(jnp.float32) + cb_ez * curl_z).astype(_fdtype)

    return state._replace(
        ex=ex, ey=ey, ez=ez,
        step=state.step + 1,
    )
