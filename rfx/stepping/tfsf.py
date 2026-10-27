"""TF/SF field corrections and full auxiliary-line steps."""

from __future__ import annotations

from rfx.core.yee import EPS_0
from rfx.core.yee import MU_0
from rfx.core.yee import MaterialArrays
from rfx.core.yee import component_h_materials
from rfx.sources.tfsf import TFSFConfig
from rfx.sources.tfsf import TFSFState
from rfx.sources.tfsf import is_tfsf_2d
from rfx.sources.tfsf import is_tfsf_methodB
from rfx.sources.tfsf import update_tfsf_1d_e
from rfx.sources.tfsf import update_tfsf_1d_h


def apply_tfsf_e(state, cfg, tfsf_st, dx: float, dt: float):
    """Apply TFSF E-field correction (call AFTER update_e).

    Corrects Ez at x_lo and x_hi+1 where the H-curl stencil
    crosses the TFSF boundary.

    Dispatches to 2D auxiliary grid version for oblique incidence.
    """
    # Dispatch to open-domain oblique Method B (real float32, 4-edge box).
    if is_tfsf_methodB(cfg):
        from rfx.stepping.tfsf_oblique import apply_methodB_e
        return apply_methodB_e(state, cfg, tfsf_st, dx, dt)
    # Dispatch to 2D if config is TFSF2DConfig
    if is_tfsf_2d(cfg):
        from rfx.sources.tfsf_2d import apply_tfsf_2d_e
        return apply_tfsf_2d_e(state, cfg, tfsf_st, dx, dt)
    if getattr(cfg, "closed_box", False):
        from rfx.sources.tfsf_discrete_plane_wave import apply_tables
        return apply_tables(state, cfg, tfsf_st, magnetic=False)
    i0 = cfg.i0
    coeff_lo = dt / (EPS_0 * cfg.e_face_widths[0])
    coeff_hi = dt / (EPS_0 * cfg.e_face_widths[1])

    # ---- Normal incidence: direct 1D grid lookup ----
    # Hy_inc at position x_lo - 0.5 → h1d[i0 - 1]
    # (h1d[i] represents Hy at (i + 0.5) in 1D grid coords;
    #  h1d[i0-1] is at position i0 - 0.5, which maps to x_lo - 0.5 in 3D)
    h_inc_lo = tfsf_st.h1d[i0 - 1]

    # Hy_inc at position x_hi + 0.5 → h1d[i0 + (x_hi - x_lo)]
    h_inc_hi = tfsf_st.h1d[i0 + (cfg.x_hi - cfg.x_lo)]

    e_field = getattr(state, cfg.electric_component)
    # Sign flips for Ey/Hz because Ampere carries -dH/dx there.
    e_field = e_field.at[cfg.x_lo, :, :].add(-cfg.curl_sign * coeff_lo * h_inc_lo)
    e_field = e_field.at[cfg.x_hi + 1, :, :].add(cfg.curl_sign * coeff_hi * h_inc_hi)

    return state._replace(**{cfg.electric_component: e_field})


def apply_tfsf_h(state, cfg, tfsf_st, dx: float, dt: float):
    """Apply TFSF H-field correction (call AFTER update_h).

    Corrects Hy at x_lo-1 and x_hi where the E-curl stencil
    crosses the TFSF boundary.

    Dispatches to 2D auxiliary grid version for oblique incidence.
    """
    mu_abs = component_h_materials(MaterialArrays(None, None, 1.0), high_wall=(False, False, False))[0] * MU_0
    # Dispatch to open-domain oblique Method B (real float32, 4-edge box).
    if is_tfsf_methodB(cfg):
        from rfx.stepping.tfsf_oblique import apply_methodB_h
        return apply_methodB_h(state, cfg, tfsf_st, dx, dt)
    # Dispatch to 2D if config is TFSF2DConfig
    if is_tfsf_2d(cfg):
        from rfx.sources.tfsf_2d import apply_tfsf_2d_h
        return apply_tfsf_2d_h(state, cfg, tfsf_st, dx, dt)
    if getattr(cfg, "closed_box", False):
        from rfx.sources.tfsf_discrete_plane_wave import apply_tables
        return apply_tables(state, cfg, tfsf_st, magnetic=True)
    i0 = cfg.i0
    coeff_lo = dt / (mu_abs * cfg.h_face_widths[0])
    coeff_hi = dt / (mu_abs * cfg.h_face_widths[1])

    # ---- Normal incidence: direct 1D grid lookup ----
    # Ez_inc at position x_lo → e1d[i0]
    e_inc_lo = tfsf_st.e1d[i0]

    # Ez_inc at position x_hi+1 → e1d[i0 + (x_hi+1 - x_lo)]
    e_inc_hi = tfsf_st.e1d[i0 + (cfg.x_hi + 1 - cfg.x_lo)]

    h_field = getattr(state, cfg.magnetic_component)
    # Sign flips for Ey/Hz because Faraday carries -dE/dx there.
    h_field = h_field.at[cfg.x_lo - 1, :, :].add(-cfg.curl_sign * coeff_lo * e_inc_lo)
    h_field = h_field.at[cfg.x_hi, :, :].add(cfg.curl_sign * coeff_hi * e_inc_hi)

    return state._replace(**{cfg.magnetic_component: h_field})


def update_tfsf_1d(cfg: TFSFConfig, st: TFSFState, dx: float, dt: float,
                   t: float) -> TFSFState:
    """Advance the 1D auxiliary FDTD by one full timestep (convenience wrapper).

    Calls update_tfsf_1d_h then update_tfsf_1d_e. For correct leapfrog
    interleaving with the 3D grid, use the split functions directly:

        loop:
          1. update_h(3D)
          2. apply_tfsf_h(3D)        ← uses e1d at time n
          3. update_tfsf_1d_h(1D)    ← h1d → time n+1/2
          4. update_e(3D)
          5. apply_tfsf_e(3D)        ← uses h1d at time n+1/2
          6. update_tfsf_1d_e(1D, t) ← e1d → time n+1, inject source
    """
    st = update_tfsf_1d_h(cfg, st, dx, dt)
    st = update_tfsf_1d_e(cfg, st, dx, dt, t)
    return st
