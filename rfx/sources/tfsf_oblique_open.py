"""Open-domain oblique TFSF plane-wave source — Method B (JAX-scan-compatible).

This is the ``jnp``/functional reimplementation of a standalone numpy prototype
(``oblique_tfsf_methodB_v2.py``, not committed).  Method B injects an oblique
plane wave into an
open (CPML) 2.5-D domain (``k̂`` in the xy-plane, thin-z periodic) using a
**dispersion-matched 1D auxiliary FDTD along k̂** plus a **4-edge TFSF box**
correction.  The incident field is a genuine discrete 1D FDTD solution whose
cell size ``d_aux`` is chosen so its numerical phase velocity matches the 3D
grid's along k̂ (Taflove Ch. 5.6 / Schneider) — so the boundary cancellation is
clean and the TFSF leakage is low.

Design invariant (Stage-2 relevant): ``MethodBConfig`` is a NEW ``NamedTuple``,
NOT a ``TFSF2DConfig`` subclass, so ``is_tfsf_2d(cfg)`` is ``False`` — the
oblique-Bloch complex64/NTFF-reject lane never fires and fields stay real
``float32``.

STAGE 1 (this module + the standalone driver) only: the per-step kernels here
are ``jnp`` + functional (no numpy in-place, no python-data-dependent indexing;
box indices are static Python ints; gather bases/weights are precomputed jnp
arrays), so they drop straight into ``lax.scan`` in Stage 2.  This module does
NOT touch rfx's scan / simulation.py.

ONE deliberate deviation from the prototype: the transverse face extents are
INCLUSIVE (``[y_lo, y_hi]`` / ``[x_lo, x_hi]``), not the prototype's exclusive
``yl:yh`` / ``xl:xh``.  The exclusive slicing left the box corner nodes at
``x_hi`` / ``y_hi`` uncorrected — 8 nodes per z-plane acting as coherent
spurious line sources — and was itself the prototype's -37..-41 dB leakage
floor (PR #461 review, F1).  With inclusive slices the through-run leakage
drops to the linear-interp gather error (-59..-132 dB, exact cancellation at
the symmetric 45 deg case); the injection angle is unchanged.

Ported sign/staggering/timing verbatim from the prototype (the authoritative
physics source):
  * H_inc decomposition ``Hy_inc = +cosθ·h1d``, ``Hx_inc = -sinθ·h1d`` — enters
    the E-face corrections as the ``±ce·ct`` / ``±ce·st`` factors below.
  * H-face (Hy/Hx) corrections read Ez_inc (``e1d``) with NO angle factor.
  * full-z-plane injection (``[:, None]`` broadcast) — the wave is z-uniform.
  * n / n+1/2 leapfrog interleave (H-face reads e1d at n, E-face reads h1d at
    n+1/2); 1D CPML (not a hard end-damper).
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from rfx.core.yee import EPS_0, MU_0
from rfx.sources.tfsf import TFSFState, update_tfsf_1d_h, update_tfsf_1d_e

C0 = 299_792_458.0

# The 1D-aux state shape is exactly the normal-path 1D FDTD state, so the reused
# update_tfsf_1d_h/e kernels advance it unchanged.
MethodBState = TFSFState


class MethodBConfig(NamedTuple):
    """Static Method-B configuration (compile-time constants for JIT/scan).

    Box indices are plain Python ``int`` (compile-time slice bounds); projection
    scalars are Python ``float`` baked at trace time; only ``b_cpml``/``c_cpml``
    and the 24 gather-table arrays are ``jnp`` (built once at init).
    """
    # --- static TFSF box indices (Python int -> compile-time slice bounds) ---
    # Total-field region is INCLUSIVE: x in [x_lo, x_hi], y in [y_lo, y_hi] —
    # the upper-face corrections (Hy[x_hi], Hx[.,y_hi], Ez[x_hi+1], Ez[.,y_hi+1])
    # are only self-consistent if x_hi / y_hi are total-field nodes, so the
    # transverse face slices must include them (see module docstring: the
    # prototype's exclusive slicing skipped the corners and leaked).
    # Corrections also touch x_lo-1, x_hi, y_lo-1, y_hi, x_hi+1, y_hi+1.
    x_lo: int
    x_hi: int
    y_lo: int
    y_hi: int
    # --- projection scalars (Python float) ---
    cos_theta: float
    sin_theta: float
    d_aux: float           # dispersion-matched 1D cell size along k-hat
    dx_1d: float           # == d_aux (reused 1D kernels read cfg.dx_1d)
    dt: float              # baked so advance_methodB_* need no dt arg
    angle_deg: float
    transverse_axis: str   # "y" for ez
    # --- 1D-aux source/CPML params (names match TFSFConfig so the 1D kernels
    #     update_tfsf_1d_h/e duck-type over this config unchanged) ---
    n_cpml: int
    b_cpml: jnp.ndarray
    c_cpml: jnp.ndarray
    src_idx: int
    src_amp: float
    src_t0: float
    src_tau: float
    src_fcen: float
    src_waveform: str
    curl_sign: float          # +1.0 for ez/hy
    electric_component: str   # "ez"
    magnetic_component: str   # "hy"
    direction: str
    direction_sign: float
    # --- 8 precomputed face-projection gather tables (jnp, static geometry) ---
    #   naming <face>_<read>: read=e -> gather e1d (H-correction);
    #                         read=h -> gather h1d half-node (E-correction).
    #   each triple (base:int32, w0:float32, w1:float32); base clipped to
    #   [0, n_aux-2] so base+1 is always in range.
    xlo_e_base: jnp.ndarray; xlo_e_w0: jnp.ndarray; xlo_e_w1: jnp.ndarray
    xhi_e_base: jnp.ndarray; xhi_e_w0: jnp.ndarray; xhi_e_w1: jnp.ndarray
    ylo_e_base: jnp.ndarray; ylo_e_w0: jnp.ndarray; ylo_e_w1: jnp.ndarray
    yhi_e_base: jnp.ndarray; yhi_e_w0: jnp.ndarray; yhi_e_w1: jnp.ndarray
    xlo_h_base: jnp.ndarray; xlo_h_w0: jnp.ndarray; xlo_h_w1: jnp.ndarray
    xhi_h_base: jnp.ndarray; xhi_h_w0: jnp.ndarray; xhi_h_w1: jnp.ndarray
    ylo_h_base: jnp.ndarray; ylo_h_w0: jnp.ndarray; ylo_h_w1: jnp.ndarray
    yhi_h_base: jnp.ndarray; yhi_h_w0: jnp.ndarray; yhi_h_w1: jnp.ndarray


# ---------------------------------------------------------------------------
# Dispersion-matched 1D cell size (bisection, numpy — init-time only).
# Ported verbatim from the prototype (k_numerical / dx_aux_matched).
# ---------------------------------------------------------------------------

def _k_numerical(omega, theta, dt, dx):
    """2D Yee numerical wavenumber |k| at angle theta (kx=|k|cosθ, ky=|k|sinθ):
    (sin(ω dt/2)/(c dt))^2 = (sin(kx dx/2)/dx)^2 + (sin(ky dy/2)/dy)^2."""
    rhs = (np.sin(omega * dt / 2.0) / (C0 * dt)) ** 2
    ct, st = np.cos(theta), np.sin(theta)

    def f(k):
        return (np.sin(k * ct * dx / 2.0) / dx) ** 2 + (np.sin(k * st * dx / 2.0) / dx) ** 2 - rhs
    lo, hi = 1e-3, np.pi / dx * 0.999
    for _ in range(200):
        m = 0.5 * (lo + hi)
        lo, hi = (lo, m) if f(lo) * f(m) <= 0 else (m, hi)
    return 0.5 * (lo + hi)


def _dx_aux_matched(omega, k_num, dt):
    """Delta_aux s.t. a 1D grid reproduces k_num at omega:
    sin(ω dt/2)/(c dt) = sin(k_num·Δ/2)/Δ."""
    rhs = np.sin(omega * dt / 2.0) / (C0 * dt)

    def f(D):
        return np.sin(k_num * D / 2.0) / D - rhs
    lo, hi = 1e-6, 2.0 * np.pi / k_num * 0.999   # first half-period
    for _ in range(200):
        m = 0.5 * (lo + hi)
        lo, hi = (lo, m) if f(lo) * f(m) <= 0 else (m, hi)
    return 0.5 * (lo + hi)


def _gather_table(idx: np.ndarray, n_aux: int):
    """Build (base, w0, w1) linear-interp gather triple from float indices.

    Matches the prototype's per-step ``interp``: floor -> fractional weight from
    the UNCLIPPED floor -> clip base to [0, n_aux-2] (so base+1 <= n_aux-1).
    Runs in numpy at init; only the int32/float32 results cross into jnp.
    """
    fl = np.floor(idx)
    w1 = (idx - fl).astype(np.float32)
    w0 = (1.0 - w1).astype(np.float32)
    base = np.clip(fl.astype(np.int32), 0, n_aux - 2)
    return (jnp.asarray(base, dtype=jnp.int32),
            jnp.asarray(w0, dtype=jnp.float32),
            jnp.asarray(w1, dtype=jnp.float32))


def init_tfsf_methodB(
    nx: int,
    ny: int,
    dx: float,
    dt: float,
    *,
    nz: int | None = None,
    cpml_layers: int = 10,
    tfsf_margin: int = 10,
    f0: float = 5e9,
    bandwidth: float = 0.5,        # accepted for signature parity; Stage-1 uses
                                   # the prototype's fixed t0=40·dt, tau=12·dt.
    amplitude: float = 1.0,
    polarization: str = "ez",
    direction: str = "+x",
    theta_deg: float = 0.0,
) -> tuple[MethodBConfig, MethodBState]:
    """Build the Method-B config + zero 1D-aux state (open-domain oblique TFSF).

    Box offset ``off = cpml_layers + tfsf_margin`` (prototype used 20 = 10+10).
    Box spans x in ``[off, nx-off)`` and y in ``[off, ny-off)``.  ``d_aux`` is
    the dispersion-matched 1D cell size along k̂ at ``ω = 2π f0``.

    Source timing reproduces the prototype exactly: ``t0 = 40·dt``,
    ``tau = 12·dt``, modulated Gaussian at carrier ``f0``.
    """
    if polarization != "ez":
        raise NotImplementedError(
            "Method B currently supports polarization='ez' (transverse=y) only; "
            "ey (transverse=z) is future work."
        )
    if direction not in ("+x", "-x"):
        raise ValueError(f"direction must be '+x' or '-x', got {direction!r}")
    if direction == "-x":
        # direction_sign is stored but never consumed: the gather tables are
        # built with +cosθ/+sinθ projections and the 1D source sits at the lo
        # pad, so '-x' would silently inject a +x-travelling wave (review F2).
        raise NotImplementedError(
            "Method B supports direction='+x' only; '-x' is not wired into the "
            "gather tables / 1D-aux layout yet. Use direction='+x' (mirror the "
            "geometry) or the normal-incidence path."
        )
    if not abs(theta_deg) < 90.0:
        raise ValueError(f"abs(theta_deg) must be < 90, got {theta_deg}")
    # NOTE: Method B's math is valid for the full range 0 <= |theta| < 90 (theta=0
    # reduces to normal incidence with d_aux==dx). theta=0 is used by the Stage-1
    # port-validation gate. In Stage 2 the init_tfsf dispatch chooses methodB vs
    # the faster normal-incidence 1D-aux path via an explicit `method` kwarg,
    # not by rejecting theta=0 here.

    theta = np.radians(theta_deg)
    ct = float(np.cos(theta))
    st = float(np.sin(theta))
    omega = 2.0 * np.pi * f0
    eta = float(np.sqrt(MU_0 / EPS_0))

    # --- TFSF box (prototype: xl,xh = off, nx-off ; yl,yh = off, ny-off) ---
    off = int(cpml_layers + tfsf_margin)
    x_lo, x_hi = off, nx - off
    y_lo, y_hi = off, ny - off
    if x_lo <= 0 or x_hi >= nx - 1 or y_lo <= 0 or y_hi >= ny - 1 \
            or x_lo >= x_hi or y_lo >= y_hi:
        raise ValueError(
            "TFSF margin/cpml_layers too large for the grid: "
            f"nx={nx}, ny={ny}, cpml_layers={cpml_layers}, tfsf_margin={tfsf_margin}"
        )

    # --- dispersion-matched 1D cell size along k-hat ---
    k_num = _k_numerical(omega, theta, dt, dx)
    d_aux = float(_dx_aux_matched(omega, k_num, dt))

    # --- 1D-aux sizing (prototype layout) ---
    n_cpml_1d = 20
    n_src_pad = 60                       # source sits 60 cells in from the lo end
    d_max = (nx * ct + ny * st) * dx     # max projection over the (padded) domain
    n_aux = int(d_max / d_aux) + 2 * n_src_pad + 4
    src_off = n_src_pad                  # 1D index mapping projection d = 0

    # --- 1D CPML profile (verbatim from prototype / rfx init_tfsf, cell=d_aux) ---
    rho = 1.0 - np.arange(n_cpml_1d, dtype=np.float64) / max(n_cpml_1d - 1, 1)
    sigma_max = 0.8 * 4.0 / (eta * d_aux)
    sigma_prof = sigma_max * rho ** 3
    alpha_prof = 0.05 * (1.0 - rho)
    denom = sigma_prof + alpha_prof
    b_prof = np.exp(-(sigma_prof + alpha_prof) * dt / EPS_0)
    c_prof = np.where(denom > 1e-30, sigma_prof * (b_prof - 1.0) / denom, 0.0)

    # --- source params: reproduce the prototype (t0=40·dt, w=12·dt, carrier f0) ---
    src_t0 = 40.0 * dt
    src_tau = 12.0 * dt

    # --- gather tables ---------------------------------------------------------
    # aux index for an E-node projection (read by H-correction, gathers e1d):
    #   idx_E(ix, iy) = (ix*ct + iy*st)*dx / d_aux + src_off
    # aux index for an H-node (read by E-correction, gathers h1d on half-nodes):
    #   idx_H(ix, iy) = idx_E(ix, iy) - 0.5
    def idx_E(ix, iy):
        return ((ix * ct + iy * st) * dx) / d_aux + src_off

    def idx_H(ix, iy):
        return idx_E(ix, iy) - 0.5

    # INCLUSIVE transverse extents — the corners x_hi / y_hi are total-field
    # nodes and must receive their correction (module docstring, review F1).
    iy_face = np.arange(y_lo, y_hi + 1)  # x-faces vary iy over [y_lo, y_hi]
    ix_face = np.arange(x_lo, x_hi + 1)  # y-faces vary ix over [x_lo, x_hi]

    # H-face tables (gather e1d) — projection at the E-node coordinate.
    xlo_e = _gather_table(idx_E(x_lo,       iy_face), n_aux)
    xhi_e = _gather_table(idx_E(x_hi + 1,   iy_face), n_aux)
    ylo_e = _gather_table(idx_E(ix_face,    y_lo),    n_aux)
    yhi_e = _gather_table(idx_E(ix_face,    y_hi + 1), n_aux)

    # E-face tables (gather h1d, half-node) — fractional cell coord + (-0.5).
    xlo_h = _gather_table(idx_H(x_lo - 0.5, iy_face), n_aux)
    xhi_h = _gather_table(idx_H(x_hi + 0.5, iy_face), n_aux)
    ylo_h = _gather_table(idx_H(ix_face,    y_lo - 0.5), n_aux)
    yhi_h = _gather_table(idx_H(ix_face,    y_hi + 0.5), n_aux)

    fdtype = jnp.float32
    cfg = MethodBConfig(
        x_lo=x_lo, x_hi=x_hi, y_lo=y_lo, y_hi=y_hi,
        cos_theta=ct, sin_theta=st,
        d_aux=d_aux, dx_1d=d_aux, dt=float(dt),
        angle_deg=float(theta_deg), transverse_axis="y",
        n_cpml=n_cpml_1d,
        b_cpml=jnp.asarray(b_prof, dtype=fdtype),
        c_cpml=jnp.asarray(c_prof, dtype=fdtype),
        src_idx=int(src_off),
        src_amp=float(amplitude),
        src_t0=float(src_t0),
        src_tau=float(src_tau),
        src_fcen=float(f0),
        src_waveform="modulated_gaussian",
        curl_sign=1.0,
        electric_component="ez",
        magnetic_component="hy",
        direction=direction,
        direction_sign=1.0 if direction == "+x" else -1.0,
        xlo_e_base=xlo_e[0], xlo_e_w0=xlo_e[1], xlo_e_w1=xlo_e[2],
        xhi_e_base=xhi_e[0], xhi_e_w0=xhi_e[1], xhi_e_w1=xhi_e[2],
        ylo_e_base=ylo_e[0], ylo_e_w0=ylo_e[1], ylo_e_w1=ylo_e[2],
        yhi_e_base=yhi_e[0], yhi_e_w0=yhi_e[1], yhi_e_w1=yhi_e[2],
        xlo_h_base=xlo_h[0], xlo_h_w0=xlo_h[1], xlo_h_w1=xlo_h[2],
        xhi_h_base=xhi_h[0], xhi_h_w0=xhi_h[1], xhi_h_w1=xhi_h[2],
        ylo_h_base=ylo_h[0], ylo_h_w0=ylo_h[1], ylo_h_w1=ylo_h[2],
        yhi_h_base=yhi_h[0], yhi_h_w0=yhi_h[1], yhi_h_w1=yhi_h[2],
    )

    state = MethodBState(
        e1d=jnp.zeros(n_aux, dtype=fdtype),
        h1d=jnp.zeros(n_aux, dtype=fdtype),
        psi_e_lo=jnp.zeros(n_cpml_1d, dtype=fdtype),
        psi_e_hi=jnp.zeros(n_cpml_1d, dtype=fdtype),
        psi_h_lo=jnp.zeros(n_cpml_1d, dtype=fdtype),
        psi_h_hi=jnp.zeros(n_cpml_1d, dtype=fdtype),
        step=jnp.array(0, dtype=jnp.int32),
    )
    return cfg, state


# ---------------------------------------------------------------------------
# Discriminator + aux advance (thin wrappers over the byte-identical 1D kernels)
# ---------------------------------------------------------------------------

def is_tfsf_methodB(cfg) -> bool:
    """True iff cfg is a Method-B (open-domain oblique) config."""
    return isinstance(cfg, MethodBConfig)


# synonym used in the integration spec
is_open_oblique = is_tfsf_methodB


def advance_methodB_h(cfg: MethodBConfig, st: MethodBState) -> MethodBState:
    """Advance the 1D-aux H field h1d^{n-1/2} -> h1d^{n+1/2} (reads e1d at n).

    Reuses the byte-identical normal-path 1D kernel (curl_sign=+1 -> identical to
    the prototype's advance_1d_h).
    """
    return update_tfsf_1d_h(cfg, st, cfg.dx_1d, cfg.dt)


def advance_methodB_e(cfg: MethodBConfig, st: MethodBState, t: float) -> MethodBState:
    """Advance the 1D-aux E field e1d^{n} -> e1d^{n+1} + source (reads h1d at n+1/2)."""
    return update_tfsf_1d_e(cfg, st, cfg.dx_1d, cfg.dt, t)


# Spec synonyms (Stage-2 dispatch names): dx/dt from the 3D call site are
# ignored by the 1D kernels (they use cfg.dx_1d/cfg.dt).
def update_open_oblique_h(cfg, st, dx, dt):
    return update_tfsf_1d_h(cfg, st, cfg.dx_1d, cfg.dt)


def update_open_oblique_e(cfg, st, dx, dt, t):
    return update_tfsf_1d_e(cfg, st, cfg.dx_1d, cfg.dt, t)


# ---------------------------------------------------------------------------
# The genuinely new code: 4-edge TFSF box corrections onto the full z-plane.
# jnp gather (arr[base]*w0 + arr[base+1]*w1) + functional .at[...].add.
# ---------------------------------------------------------------------------

def _gather(arr1d, base, w0, w1):
    """Linear-interp gather; base+1 is safe (clipped to n_aux-2 at init)."""
    return arr1d[base] * w0 + arr1d[base + 1] * w1


def apply_methodB_h(state, cfg: MethodBConfig, st: MethodBState, dx: float, dt: float):
    """4-edge H-field TFSF correction (call AFTER update_h, reads e1d at time n).

    Reproduces the prototype (v2 lines 184-188): Hy at x_lo-1 / x_hi and Hx at
    y_lo-1 / y_hi get ``±ch·Ez_inc`` with NO cos/sin factor (the angle enters via
    the k̂ projection baked into the gather tables). Full-z broadcast ([:, None]).
    """
    ch = dt / (MU_0 * dx)
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
