"""Open-domain oblique TF/SF auxiliary advances and field corrections."""

from __future__ import annotations

from rfx.core.yee import EPS_0
from rfx.core.yee import MU_0
from rfx.sources.tfsf import update_tfsf_1d_e
from rfx.sources.tfsf import update_tfsf_1d_h
from rfx.sources.tfsf_oblique_open import MethodBConfig
from rfx.sources.tfsf_oblique_open import MethodBState


def advance_methodB_e(cfg: MethodBConfig, st: MethodBState, t: float) -> MethodBState:
    """Advance the 1D-aux E field e1d^{n} -> e1d^{n+1} + source (reads h1d at n+1/2)."""
    return update_tfsf_1d_e(cfg, st, cfg.dx_1d, cfg.dt, t)


def advance_methodB_h(cfg: MethodBConfig, st: MethodBState) -> MethodBState:
    """Advance the 1D-aux H field h1d^{n-1/2} -> h1d^{n+1/2} (reads e1d at n).

    Reuses the byte-identical normal-path 1D kernel (curl_sign=+1 -> identical to
    the prototype's advance_1d_h).
    """
    return update_tfsf_1d_h(cfg, st, cfg.dx_1d, cfg.dt)


def apply_methodB_e(state, cfg: MethodBConfig, st: MethodBState, dx: float, dt: float):
    """4-edge E-field TFSF correction (call AFTER update_e, reads h1d at n+1/2).

    Reproduces the prototype (v2 lines 210-213). H_inc decomposition
    ``Hy_inc = +cosθ·h1d``, ``Hx_inc = -sinθ·h1d`` enters as the ``±ce·ct`` /
    ``±ce·st`` factors. h1d is read on half-nodes (idx_E - 0.5, baked in tables).
    Full-z broadcast ([:, None]).
    """
    ce = dt / (EPS_0 * dx)
    ct, sn = cfg.cos_theta, cfg.sin_theta
    h1d = st.h1d
    ez = state.ez

    # x-faces: Ez sees d(Hy)/dx ; Hy_inc = +cosθ·h1d
    inc = _gather(h1d, cfg.xlo_h_base, cfg.xlo_h_w0, cfg.xlo_h_w1)
    ez = ez.at[cfg.x_lo, cfg.y_lo:cfg.y_hi + 1, :].add((-ce * ct * inc)[:, None])
    inc = _gather(h1d, cfg.xhi_h_base, cfg.xhi_h_w0, cfg.xhi_h_w1)
    ez = ez.at[cfg.x_hi + 1, cfg.y_lo:cfg.y_hi + 1, :].add((ce * ct * inc)[:, None])

    # y-faces: Ez sees d(Hx)/dy ; Hx_inc = -sinθ·h1d
    inc = _gather(h1d, cfg.ylo_h_base, cfg.ylo_h_w0, cfg.ylo_h_w1)
    ez = ez.at[cfg.x_lo:cfg.x_hi + 1, cfg.y_lo, :].add((-ce * sn * inc)[:, None])
    inc = _gather(h1d, cfg.yhi_h_base, cfg.yhi_h_w0, cfg.yhi_h_w1)
    ez = ez.at[cfg.x_lo:cfg.x_hi + 1, cfg.y_hi + 1, :].add((ce * sn * inc)[:, None])

    return state._replace(ez=ez)


def apply_methodB_h(state, cfg: MethodBConfig, st: MethodBState, dx: float, dt: float):
    """4-edge H-field TFSF correction (call AFTER update_h, reads e1d at time n).

    Reproduces the prototype (v2 lines 184-188): Hy at x_lo-1 / x_hi and Hx at
    y_lo-1 / y_hi get ``±ch·Ez_inc`` with NO cos/sin factor (the angle enters via
    the k̂ projection baked into the gather tables). Full-z broadcast ([:, None]).
    """
    from rfx.core.yee import MaterialArrays, component_h_materials
    mu_abs = component_h_materials(MaterialArrays(None, None, 1.0), high_wall=(False, False, False))[0] * MU_0
    ch = dt / (mu_abs * dx)
    e1d = st.e1d
    hx, hy = state.hx, state.hy

    # x-faces: Hy from Ez_inc (E-node projection at x_lo / x_hi+1)
    inc = _gather(e1d, cfg.xlo_e_base, cfg.xlo_e_w0, cfg.xlo_e_w1)
    hy = hy.at[cfg.x_lo - 1, cfg.y_lo:cfg.y_hi + 1, :].add((-ch * inc)[:, None])
    inc = _gather(e1d, cfg.xhi_e_base, cfg.xhi_e_w0, cfg.xhi_e_w1)
    hy = hy.at[cfg.x_hi, cfg.y_lo:cfg.y_hi + 1, :].add((ch * inc)[:, None])

    # y-faces: Hx from Ez_inc (E-node projection at y_lo / y_hi+1)
    inc = _gather(e1d, cfg.ylo_e_base, cfg.ylo_e_w0, cfg.ylo_e_w1)
    hx = hx.at[cfg.x_lo:cfg.x_hi + 1, cfg.y_lo - 1, :].add((ch * inc)[:, None])
    inc = _gather(e1d, cfg.yhi_e_base, cfg.yhi_e_w0, cfg.yhi_e_w1)
    hx = hx.at[cfg.x_lo:cfg.x_hi + 1, cfg.y_hi, :].add((-ch * inc)[:, None])

    return state._replace(hx=hx, hy=hy)


def update_open_oblique_e(cfg, st, dx, dt, t):
    return update_tfsf_1d_e(cfg, st, cfg.dx_1d, cfg.dt, t)


def update_open_oblique_h(cfg, st, dx, dt):
    return update_tfsf_1d_h(cfg, st, cfg.dx_1d, cfg.dt)


def _gather(arr1d, base, w0, w1):
    """Linear-interp gather; base+1 is safe (clipped to n_aux-2 at init)."""
    return arr1d[base] * w0 + arr1d[base + 1] * w1
