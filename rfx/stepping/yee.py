"""Yee time-step kernels, operating on explicit field and material arrays."""
from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from rfx import _realized
from rfx.core.yee import (
    FDTDState, MaterialArrays, MU_0,
    _ca_cb_inv_eps_r_units, _diff_fwd_o, _e_update_coeffs_inv_si,
    _nu_e_metric_kwargs, _require_nu_aniso_metrics, _shift_fwd,
    component_e_materials, component_h_materials, curl_h, curl_h_nu,
    e_component_coeffs, e_update_coeffs, si_value_eps_r_grad,
)


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
    if materials.mu_r_wire is not None and so != 2:
        raise NotImplementedError("wire-port radius requires the second-order Yee stencil")
    if bloch is not None and so != 2:
        raise ValueError(
            f"bloch phase (oblique-periodic BC, #404) requires stencil_order=2, got {so}"
        )

    # Compute dtype: the oblique-periodic Bloch path carries a complex field
    # envelope P; the real path computes at ``max(state dtype, float32)`` --
    # upcast for reduced-precision (float16) fields (byte-identical for
    # float32 storage), and PROMOTED for float64 storage so the arithmetic
    # does not re-quantize a higher-precision field back to float32 every
    # timestep (issue #630 -- the prior hard `jnp.float32` pin here silently
    # discarded float64 field precision regardless of storage dtype).
    _fdtype = state.ex.dtype
    _cdtype = (
        jnp.complex64 if jnp.iscomplexobj(state.ex)
        else jnp.promote_types(state.ex.dtype, jnp.float32)
    )
    ex = state.ex.astype(_cdtype)
    ey = state.ey.astype(_cdtype)
    ez = state.ez.astype(_cdtype)
    mu = (materials.components.mu_update if materials.components is not None
          else component_h_materials(materials, periodic))
    if _realized.ACTIVE is not None:
        mu = _realized.magnetic(materials, mu, "yee.H", periodic=periodic)
    mu_x, mu_y, mu_z = (m * MU_0 for m in mu)
    ch_x, ch_y, ch_z = dt / mu_x, dt / mu_y, dt / mu_z

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

    hx = (state.hx.astype(_cdtype) - ch_x * curl_x).astype(_fdtype)
    hy = (state.hy.astype(_cdtype) - ch_y * curl_y).astype(_fdtype)
    hz = (state.hz.astype(_cdtype) - ch_z * curl_z).astype(_fdtype)

    return state._replace(hx=hx, hy=hy, hz=hz)


def update_e_box(state: FDTDState, prev: FDTDState, box: tuple,
                 ca, cb, dx: float,
                 periodic: tuple = (False, False, False),
                 stencil_order: int = 2,
                 bloch: tuple | None = None,
                 inv_d: tuple | None = None, *, boundary=None) -> FDTDState:
    """Redo the E update inside one index box with its own Ca/Cb (#1179).

    ``prev`` is the state BEFORE the grid-wide E update of this timestep. It
    carries the same H^{n+1/2} that update consumed (nothing between the two
    touches H), so this recomputes ``E^{n+1} = Ca·E^n + Cb·curl(H^{n+1/2})``
    at the box cells exactly as :func:`update_e` would — from coefficients
    that do not have to come from ``materials``.

    Why it exists: the Yee E update is LINEAR in the fields, so reverse-mode
    AD needs the primal field only where a coefficient is a design variable.
    With ``materials`` left constant and only this box recomputed from a
    traced permittivity, every array the backward pass has to keep is
    box-shaped — ``prev.E`` and ``curl(H)`` enter only through the sliced
    products, and slicing and scatter are both linear (they save nothing).
    The grid-sized alternative is a traced ``materials.eps_r``, which puts
    six grid-sized arrays per step on the tape.

    ``ca`` and ``cb`` are box-shaped (or scalar) and built once, outside the
    time loop, by :func:`e_update_coeffs`. Each may instead be a 3-TUPLE
    (or list) of box-shaped arrays, one per E component in x, y, z order:
    component ``c`` is then updated with ``ca[c]``/``cb[c]``. Two things
    produce that form. The #1210 edge average makes every coefficient
    per-component — an E component takes the mean of eps and sigma over the
    four cells its edge touches — which is what
    :func:`rfx.model.materials._design_box_edge_coeffs` builds. And a conducting
    SHEET (#1216) gives each component its own conductivity: its current runs
    along the two in-plane edges only, so the third component must keep the
    background coefficient while the other two carry the design conductivity.
    ``(v, v, v)`` is the scalar/array form cell for cell. Both sequence types
    are accepted here because :func:`rfx.model.resolve_design_box`
    accepts either for ``DesignBoxSpec.sigma``; it normalises what it builds
    to a tuple, so a list reaches this kernel only from a direct call.

    ``inv_d`` selects the GRADED-MESH curl (#1183). ``None`` (default) is the
    uniform :func:`curl_h` at spacing ``dx``; a tuple
    ``(inv_dx, inv_dy, inv_dz)`` of the E-update inverse spacings is
    :func:`curl_h_nu` instead, the curl :func:`update_e_nu` uses — so the box
    reproduces the non-uniform update the same way it reproduces the uniform
    one. ``dx``, ``periodic``, ``stencil_order`` and ``bloch`` are then
    unused: ``boundary`` supplies periodic flags to the graded curl,
    which has no fourth-order stencil or Bloch phase.

    Cost: one extra whole-grid ``curl_h`` per step, the same shared stencil
    helper the E update and the #677 sheet operator use. Slicing H to the box
    plus a halo would avoid it, at the price of a second spelling of the
    stencil; measured on CPU the extra curl is repaid by the smaller tape.
    """
    i0, i1, j0, j1, k0, k1 = box
    sl = (slice(i0, i1), slice(j0, j1), slice(k0, k1))

    # Same compute/storage dtype policy as update_e -- the two results are
    # compared cell-for-cell by the equality gate, so they must round alike.
    # update_e_nu's policy is the same expression without the complex branch
    # (it has no Bloch path), so one spelling serves both lanes.
    _fdtype = state.ex.dtype
    _cdtype = (
        jnp.complex64 if jnp.iscomplexobj(state.ex)
        else jnp.promote_types(state.ex.dtype, jnp.float32)
    )
    if inv_d is None:
        curl_x, curl_y, curl_z = curl_h(
            prev.hx.astype(_cdtype), prev.hy.astype(_cdtype),
            prev.hz.astype(_cdtype), dx, periodic, stencil_order, bloch,
            boundary=boundary)
    else:
        curl_x, curl_y, curl_z = curl_h_nu(
            prev.hx.astype(_cdtype), prev.hy.astype(_cdtype),
            prev.hz.astype(_cdtype), *inv_d, boundary=boundary)

    def _comp(v, c):
        """``v`` for every component, or the c-th entry of a 3-sequence."""
        return v[c] if isinstance(v, (tuple, list)) else v

    ex = (_comp(ca, 0) * prev.ex[sl].astype(_cdtype)
          + _comp(cb, 0) * curl_x[sl]).astype(_fdtype)
    ey = (_comp(ca, 1) * prev.ey[sl].astype(_cdtype)
          + _comp(cb, 1) * curl_y[sl]).astype(_fdtype)
    ez = (_comp(ca, 2) * prev.ez[sl].astype(_cdtype)
          + _comp(cb, 2) * curl_z[sl]).astype(_fdtype)

    return state._replace(
        ex=state.ex.at[sl].set(ex),
        ey=state.ey.at[sl].set(ey),
        ez=state.ez.at[sl].set(ez),
    )


@partial(jax.jit, static_argnames=("periodic", "stencil_order", "bloch", "boundary"))
def update_e(state: FDTDState, materials: MaterialArrays, dt: float, dx: float,
             periodic: tuple = (False, False, False),
             stencil_order: int = 2,
             bloch: tuple | None = None, *, boundary=None) -> FDTDState:
    """Electric field full-step update (Ampere's law).

    For lossy media with conductivity σ:
    E^{n+1} = Ca * E^n + Cb * curl(H^{n+1/2})

    Ca = (1 - σ*dt/(2ε)) / (1 + σ*dt/(2ε))
    Cb = (dt/ε) / (1 + σ*dt/(2ε))

    ε and σ are per COMPONENT (#1210): a Yee E component sits on an edge
    shared by four cells, and takes the mean of their ε and σ — see
    :func:`edge_averaged_materials` for the physics and the boundary
    convention. Homogeneous regions are bit-identical to the cell-owned rule
    this replaces.

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

    # Compute dtype: complex for the Bloch envelope path, ``max(state dtype,
    # float32)`` for the real path (byte-identical for float32 storage;
    # upcast covers reduced-precision fields; promoted for float64 storage
    # -- see the matching comment in update_h, issue #630).
    _fdtype = state.ex.dtype
    _cdtype = (
        jnp.complex64 if jnp.iscomplexobj(state.ex)
        else jnp.promote_types(state.ex.dtype, jnp.float32)
    )
    hx = state.hx.astype(_cdtype)
    hy = state.hy.astype(_cdtype)
    hz = state.hz.astype(_cdtype)
    # #1210: the coefficients are per COMPONENT, from the mean of eps_r and
    # sigma over the four cells incident to that component's edge.
    (ca_x, ca_y, ca_z), (cb_x, cb_y, cb_z) = e_component_coeffs(
        materials, dt, periodic)

    curl_x, curl_y, curl_z = curl_h(hx, hy, hz, dx, periodic, so, bloch, boundary=boundary)

    ex = (ca_x * state.ex.astype(_cdtype) + cb_x * curl_x).astype(_fdtype)
    ey = (ca_y * state.ey.astype(_cdtype) + cb_y * curl_y).astype(_fdtype)
    ez = (ca_z * state.ez.astype(_cdtype) + cb_z * curl_z).astype(_fdtype)

    return state._replace(
        ex=ex, ey=ey, ez=ez,
        step=state.step + 1,
    )


def update_h_nu(state: FDTDState, materials: MaterialArrays, dt: float,
                inv_dx_h: jnp.ndarray, inv_dy_h: jnp.ndarray, inv_dz_h: jnp.ndarray,
                *, cell_sizes=None, periodic=(False, False, False),
                ) -> FDTDState:
    """H update for non-uniform Yee grid.

    The H-curl differences two E nodes that straddle a single cell, so
    it divides by the LOCAL cell width (CORE-C2 fix, 2026-05-16).

    Parameters
    ----------
    inv_dx_h, inv_dy_h, inv_dz_h : (N,) arrays
        H-update inverse spacing ``inv_d_h[k] = 1/d[k]`` for k<N-1,
        ``inv_d_h[N-1] = 0`` on non-periodic axes; periodic axes use 1/d
        everywhere. Built by
        ``rfx.nonuniform._profile_to_inv_arrays`` (second return value).
    """
    # Compute dtype: max(state dtype, float32) -- byte-identical for float32
    # storage, upcast for float16, promoted for float64 (issue #630).
    _fdtype = state.ex.dtype
    _cdtype = jnp.promote_types(_fdtype, jnp.float32)
    ex = state.ex.astype(_cdtype)
    ey = state.ey.astype(_cdtype)
    ez = state.ez.astype(_cdtype)
    if cell_sizes is None:
        # Compatibility for direct kernel callers supplying inverse primal
        # metrics. Production grids pass their original cell lengths.
        cell_sizes = tuple(1 / jnp.where(v > 0, v, v[-2])
                           for v in (inv_dx_h, inv_dy_h, inv_dz_h))
    mu = (materials.components.mu_update
          if materials.components is not None
          else component_h_materials(materials, periodic, cell_sizes=cell_sizes))
    if _realized.ACTIVE is not None:
        mu = _realized.magnetic(materials, mu, "yee.H", periodic=periodic, cell_sizes=cell_sizes)
    mu_x, mu_y, mu_z = (m * MU_0 for m in mu)
    ch_x, ch_y, ch_z = dt / mu_x, dt / mu_y, dt / mu_z

    # Forward differences wrap periodic axes; other axes keep zero padding.
    curl_x = (
        _diff_fwd_o(ez, 1, periodic, 2) * inv_dy_h[None, :, None]
        - _diff_fwd_o(ey, 2, periodic, 2) * inv_dz_h[None, None, :]
    )
    curl_y = (
        _diff_fwd_o(ex, 2, periodic, 2) * inv_dz_h[None, None, :]
        - _diff_fwd_o(ez, 0, periodic, 2) * inv_dx_h[:, None, None]
    )
    curl_z = (
        _diff_fwd_o(ey, 0, periodic, 2) * inv_dx_h[:, None, None]
        - _diff_fwd_o(ex, 1, periodic, 2) * inv_dy_h[None, :, None]
    )

    hx = (state.hx.astype(_cdtype) - ch_x * curl_x).astype(_fdtype)
    hy = (state.hy.astype(_cdtype) - ch_y * curl_y).astype(_fdtype)
    hz = (state.hz.astype(_cdtype) - ch_z * curl_z).astype(_fdtype)

    return state._replace(hx=hx, hy=hy, hz=hz)


def update_e_nu(state: FDTDState, materials: MaterialArrays, dt: float,
                inv_dx: jnp.ndarray, inv_dy: jnp.ndarray, inv_dz: jnp.ndarray,
                *, boundary=None, cell_sizes=None) -> FDTDState:
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
    # Compute dtype: max(state dtype, float32) -- byte-identical for float32
    # storage, upcast for float16, promoted for float64 (issue #630).
    _fdtype = state.ex.dtype
    _cdtype = jnp.promote_types(_fdtype, jnp.float32)
    hx = state.hx.astype(_cdtype)
    hy = state.hy.astype(_cdtype)
    hz = state.hz.astype(_cdtype)

    # One spelling with the uniform lane and with update_e_box, so a design
    # box on a graded mesh builds the SAME Ca/Cb from its own permittivity
    # (#1183). #1210: per component, from the mean over the four cells
    # incident to the edge, with periodic flags from the NU boundary.
    (ca_x, ca_y, ca_z), (cb_x, cb_y, cb_z) = e_component_coeffs(
        materials, dt, boundary.periodic if boundary is not None else (False,) * 3,
        **_nu_e_metric_kwargs(materials, (inv_dx, inv_dy, inv_dz), cell_sizes))

    # Backward differences with same shape (zero-pad via _shift_bwd)
    curl_x, curl_y, curl_z = curl_h_nu(hx, hy, hz, inv_dx, inv_dy, inv_dz, boundary=boundary)

    ex = (ca_x * state.ex.astype(_cdtype) + cb_x * curl_x).astype(_fdtype)
    ey = (ca_y * state.ey.astype(_cdtype) + cb_y * curl_y).astype(_fdtype)
    ez = (ca_z * state.ez.astype(_cdtype) + cb_z * curl_z).astype(_fdtype)

    return state._replace(ex=ex, ey=ey, ez=ez, step=state.step + 1)


def update_h_fast(state: FDTDState, ch: jnp.ndarray | tuple) -> FDTDState:
    """H update using pre-computed coefficient ``ch = dt/(mu*dx)``.

    Avoids recomputing material coefficients each timestep.
    Non-periodic boundaries only (uses ``_shift_fwd``).
    """
    # Compute dtype: max(state dtype, float32) -- byte-identical for float32
    # storage, upcast for float16, promoted for float64 (issue #630).
    _fdtype = state.ex.dtype
    _cdtype = jnp.promote_types(_fdtype, jnp.float32)
    ex = state.ex.astype(_cdtype)
    ey = state.ey.astype(_cdtype)
    ez = state.ez.astype(_cdtype)
    cx, cy, cz = ch if isinstance(ch, tuple) else (ch,) * 3
    hx = (state.hx.astype(_cdtype) - cx * ((_shift_fwd(ez, 1) - ez) - (_shift_fwd(ey, 2) - ey))).astype(_fdtype)
    hy = (state.hy.astype(_cdtype) - cy * ((_shift_fwd(ex, 2) - ex) - (_shift_fwd(ez, 0) - ez))).astype(_fdtype)
    hz = (state.hz.astype(_cdtype) - cz * ((_shift_fwd(ey, 0) - ey) - (_shift_fwd(ex, 1) - ex))).astype(_fdtype)
    return state._replace(hx=hx, hy=hy, hz=hz)


def update_e_fast(
    state: FDTDState,
    ca_ex: jnp.ndarray, ca_ey: jnp.ndarray, ca_ez: jnp.ndarray,
    cb_ex: jnp.ndarray, cb_ey: jnp.ndarray, cb_ez: jnp.ndarray,
    *, boundary=None,
) -> FDTDState:
    """E update using pre-computed per-component coefficients.

    ``ca_*`` and ``cb_*`` already include PEC zeroing when built with
    :func:`precompute_coeffs` and ``pec_axes``, so no separate
    ``apply_pec()`` call is needed.

    Non-periodic boundaries only (uses ``_shift_bwd``).
    """
    # Compute dtype: max(state dtype, float32) -- byte-identical for float32
    # storage, upcast for float16, promoted for float64 (issue #630).
    _fdtype = state.ex.dtype
    _cdtype = jnp.promote_types(_fdtype, jnp.float32)
    hx = state.hx.astype(_cdtype)
    hy = state.hy.astype(_cdtype)
    hz = state.hz.astype(_cdtype)
    curl_x, curl_y, curl_z = curl_h(
        hx, hy, hz, None, (False, False, False), boundary=boundary)
    ex = (ca_ex * state.ex.astype(_cdtype) + cb_ex * curl_x).astype(_fdtype)
    ey = (ca_ey * state.ey.astype(_cdtype) + cb_ey * curl_y).astype(_fdtype)
    ez = (ca_ez * state.ez.astype(_cdtype) + cb_ez * curl_z).astype(_fdtype)
    return state._replace(ex=ex, ey=ey, ez=ez, step=state.step + 1)


@_require_nu_aniso_metrics
@partial(jax.jit, static_argnames=("boundary",))
def update_e_nu_aniso(state: FDTDState, materials: MaterialArrays,
                      eps_ex: jnp.ndarray, eps_ey: jnp.ndarray, eps_ez: jnp.ndarray,
                      dt: float,
                      inv_dx: jnp.ndarray, inv_dy: jnp.ndarray, inv_dz: jnp.ndarray,
                      *, boundary=None, cell_sizes=None) -> FDTDState:
    """Non-uniform E update with per-component anisotropic permittivity.

    Same backward-difference structure as :func:`update_e_nu` (E-update
    mean spacing via ``inv_dx``/``inv_dy``/``inv_dz``) but uses three separate
    permittivity arrays (Ex, Ey, Ez) so subpixel-smoothing can be applied
    on a non-uniform mesh. ``materials.sigma`` is still applied
    isotropically (matches the uniform-path :func:`update_e_aniso`).
    With ``subpixel_smoothing="full_tensor"`` the node tensor's
    off-diagonal increment is added as on the uniform path, its couplings
    weighted by the local cell sizes (#1632).
    """
    # Compute dtype: max(state dtype, float32) -- byte-identical for float32
    # storage, upcast for float16, promoted for float64 (issue #630).
    _fdtype = state.ex.dtype
    _cdtype = jnp.promote_types(_fdtype, jnp.float32)
    hx = state.hx.astype(_cdtype)
    hy = state.hy.astype(_cdtype)
    hz = state.hz.astype(_cdtype)
    # #1210: sigma is a VOLUME conductivity, so it takes the same
    # per-component edge average the subpixel lane already gives eps -- the
    # conduction paths of the four cells lie in parallel along the edge. The
    # permittivity here stays the Kottke/subpixel tensor, which is already
    # per-component by construction. Where sigma is uniform (every lossless
    # subpixel fixture) the mean of four equal floats is that float, so those
    # runs keep their bytes. Periodic axes use the boundary's seam average.
    sigma_ex, sigma_ey, sigma_ez = (materials.components.sigma_update if materials.components is not None
        else component_e_materials(materials, boundary.periodic if boundary is not None else (False,) * 3,
            **({} if cell_sizes is None else dict(cell_sizes=cell_sizes)))[1])

    # The same arithmetic as update_e's, so one spelling (and its #1357
    # eps_r-unit derivative) serves both.
    ca_ex, cb_ex = e_update_coeffs(eps_ex, sigma_ex, dt)
    ca_ey, cb_ey = e_update_coeffs(eps_ey, sigma_ey, dt)
    ca_ez, cb_ez = e_update_coeffs(eps_ez, sigma_ez, dt)

    # Backward differences with per-cell inv-spacing (mirrors update_e_nu)
    curl_x, curl_y, curl_z = curl_h_nu(hx, hy, hz, inv_dx, inv_dy, inv_dz, boundary=boundary)

    ex = (ca_ex * state.ex.astype(_cdtype) + cb_ex * curl_x).astype(_fdtype)
    ey = (ca_ey * state.ey.astype(_cdtype) + cb_ey * curl_y).astype(_fdtype)
    ez = (ca_ez * state.ez.astype(_cdtype) + cb_ez * curl_z).astype(_fdtype)

    tensor = materials.components.tensor if materials.components is not None else None
    if tensor is not None and tensor.node_smoothing is not None:
        from rfx.core.node_tensor import add_node_increment
        ex, ey, ez = add_node_increment(
            (ex, ey, ez), (curl_x, curl_y, curl_z), tensor.node_smoothing,
            (eps_ex, eps_ey, eps_ez), (sigma_ex, sigma_ey, sigma_ez), dt,
            periodic=boundary.periodic if boundary is not None else (False,) * 3,
            boundary=boundary)

    return state._replace(ex=ex, ey=ey, ez=ez, step=state.step + 1)


@partial(jax.jit, static_argnames=("periodic", "boundary"))
def update_e_aniso_inv(state: FDTDState, materials: MaterialArrays,
                       inv_xx: jnp.ndarray, inv_yy: jnp.ndarray, inv_zz: jnp.ndarray,
                       dt: float, dx: float,
                       periodic: tuple = (False, False, False), *, boundary=None) -> FDTDState:
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
    # Compute dtype: max(state dtype, float32) -- byte-identical for float32
    # storage, upcast for float16, promoted for float64 (issue #630).
    _fdtype = state.ex.dtype
    _cdtype = jnp.promote_types(_fdtype, jnp.float32)
    hx = state.hx.astype(_cdtype)
    hy = state.hy.astype(_cdtype)
    hz = state.hz.astype(_cdtype)
    # #1210: sigma is a VOLUME conductivity, so it takes the same
    # per-component edge average the subpixel lane already gives eps -- the
    # conduction paths of the four cells lie in parallel along the edge. The
    # permittivity here stays the Kottke/subpixel tensor, which is already
    # per-component by construction. Where sigma is uniform (every lossless
    # subpixel fixture) the mean of four equal floats is that float, so those
    # runs keep their bytes.
    sigma_ex, sigma_ey, sigma_ez = (materials.components.sigma_update if materials.components is not None
        else component_e_materials(materials, periodic)[1])

    # Per-component lossy update coefficients in inv-eps form
    # (:func:`_e_update_coeffs_inv_si`). #1357: those bits, the eps_r-unit
    # derivative (``* inv_eps0`` handed the cotangent a factor 1/EPS_0).
    ca_ex, cb_ex = si_value_eps_r_grad(
        _e_update_coeffs_inv_si, _ca_cb_inv_eps_r_units, inv_xx, sigma_ex, dt)
    ca_ey, cb_ey = si_value_eps_r_grad(
        _e_update_coeffs_inv_si, _ca_cb_inv_eps_r_units, inv_yy, sigma_ey, dt)
    ca_ez, cb_ez = si_value_eps_r_grad(
        _e_update_coeffs_inv_si, _ca_cb_inv_eps_r_units, inv_zz, sigma_ez, dt)

    # curl H (identical to update_e and update_e_aniso).
    curl_x, curl_y, curl_z = curl_h(hx, hy, hz, dx, periodic, boundary=boundary)

    ex = (ca_ex * state.ex.astype(_cdtype) + cb_ex * curl_x).astype(_fdtype)
    ey = (ca_ey * state.ey.astype(_cdtype) + cb_ey * curl_y).astype(_fdtype)
    ez = (ca_ez * state.ez.astype(_cdtype) + cb_ez * curl_z).astype(_fdtype)

    return state._replace(
        ex=ex, ey=ey, ez=ez,
        step=state.step + 1,
    )


def update_e_aniso(state: FDTDState, materials: MaterialArrays,
                   eps_ex: jnp.ndarray, eps_ey: jnp.ndarray, eps_ez: jnp.ndarray,
                   dt: float, dx: float,
                   periodic: tuple = (False, False, False), *, boundary=None) -> FDTDState:
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
    # Compute dtype: max(state dtype, float32) -- byte-identical for float32
    # storage, upcast for float16, promoted for float64 (issue #630).
    _fdtype = state.ex.dtype
    _cdtype = jnp.promote_types(_fdtype, jnp.float32)
    hx = state.hx.astype(_cdtype)
    hy = state.hy.astype(_cdtype)
    hz = state.hz.astype(_cdtype)
    # #1210: sigma is a VOLUME conductivity, so it takes the same
    # per-component edge average the subpixel lane already gives eps -- the
    # conduction paths of the four cells lie in parallel along the edge. The
    # permittivity here stays the Kottke/subpixel tensor, which is already
    # per-component by construction. Where sigma is uniform (every lossless
    # subpixel fixture) the mean of four equal floats is that float, so those
    # runs keep their bytes.
    sigma_ex, sigma_ey, sigma_ez = (materials.components.sigma_update if materials.components is not None
        else component_e_materials(materials, periodic)[1])

    if _realized.ACTIVE is not None:
        (eps_ex, eps_ey, eps_ez), (sigma_ex, sigma_ey, sigma_ez) = _realized.electric(
            materials, (eps_ex, eps_ey, eps_ez),
            (sigma_ex, sigma_ey, sigma_ez), "aniso.E", periodic=periodic)

    # Per-component lossy update coefficients: the arithmetic of
    # e_update_coeffs, so one spelling (and its #1357 eps_r-unit
    # derivative) serves both.
    ca_ex, cb_ex = e_update_coeffs(eps_ex, sigma_ex, dt)
    ca_ey, cb_ey = e_update_coeffs(eps_ey, sigma_ey, dt)
    ca_ez, cb_ez = e_update_coeffs(eps_ez, sigma_ez, dt)

    # curl H (same as update_e)
    curl_x, curl_y, curl_z = curl_h(hx, hy, hz, dx, periodic, boundary=boundary)

    ex = (ca_ex * state.ex.astype(_cdtype) + cb_ex * curl_x).astype(_fdtype)
    ey = (ca_ey * state.ey.astype(_cdtype) + cb_ey * curl_y).astype(_fdtype)
    ez = (ca_ez * state.ez.astype(_cdtype) + cb_ez * curl_z).astype(_fdtype)

    tensor = materials.components.tensor if materials.components is not None else None
    if tensor is not None and tensor.node_smoothing is not None:
        from rfx.core.node_tensor import add_node_increment
        ex, ey, ez = add_node_increment(
            (ex, ey, ez), (curl_x, curl_y, curl_z), tensor.node_smoothing,
            (eps_ex, eps_ey, eps_ez), (sigma_ex, sigma_ey, sigma_ez), dt,
            periodic=periodic, boundary=boundary)

    return state._replace(
        ex=ex, ey=ey, ez=ez,
        step=state.step + 1,
    )
