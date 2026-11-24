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
values at the selected precision (including the existing x64 promotion).

STAGE 1 (this module + the standalone driver) only: the per-step kernels here
are ``jnp`` + functional (no numpy in-place, no python-data-dependent indexing;
box indices are static Python ints; gather bases/weights are precomputed jnp
arrays), so they drop straight into ``lax.scan`` in Stage 2.  This module does
NOT touch rfx's scan / simulation.py.

The transverse face extents differ from the prototype: they are
INCLUSIVE (``[y_lo, y_hi]`` / ``[x_lo, x_hi]``), not the prototype's exclusive
``yl:yh`` / ``xl:xh``.  The exclusive slicing left the box corner nodes at
``x_hi`` / ``y_hi`` uncorrected — 8 nodes per z-plane acting as coherent
spurious line sources — and was itself the prototype's -37..-41 dB leakage
floor.  With inclusive slices the through-run leakage
drops to the linear-interp gather error (-59..-132 dB, exact cancellation at
the symmetric 45 deg case); the injection angle is unchanged.

Pulse timing follows the declared waveform through ``pulse_timing``.
Signs, staggering and leapfrog order retain the prototype physics:
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

from rfx.boundaries.cpml import CPMLParams, _cpml_profile
from rfx.core.yee import EPS_0, MU_0
from rfx.sources.tfsf import TFSFState, update_tfsf_1d_h, update_tfsf_1d_e

_METHOD_B_SOURCE_PAD_CELLS = 60


C0 = 299_792_458.0

MethodBState = TFSFState


class MethodBConfig(NamedTuple):
    """Static Method-B configuration (compile-time constants for JIT/scan).

    Box indices are plain Python ``int`` (compile-time slice bounds); projection
    scalars are Python ``float`` baked at trace time; the CPML profiles
    and the 24 gather-table arrays are ``jnp`` (built once at init).

    Each gather triple is (base int32, w0 and w1 in the coefficient dtype);
    base is clipped to [0, n_aux-2] so base+1 is always in range.

    Gather fields named ``<face>_e_<part>`` read electric nodes for magnetic corrections, while fields named ``<face>_h_<part>`` read magnetic half-nodes for electric corrections.
    """
    x_lo: int
    x_hi: int
    y_lo: int
    y_hi: int
    cos_theta: float
    sin_theta: float
    d_aux: float
    dx_1d: float
    dt: float
    angle_deg: float
    transverse_axis: str
    n_cpml: int
    b_cpml: jnp.ndarray
    c_cpml: jnp.ndarray
    magnetic_lo: CPMLParams
    magnetic_hi: CPMLParams
    src_idx: int
    src_amp: float
    src_t0: float
    src_tau: float
    src_fcen: float
    src_waveform: str
    curl_sign: float
    electric_component: str
    magnetic_component: str
    direction: str
    direction_sign: float
    xlo_e_base: jnp.ndarray; xlo_e_w0: jnp.ndarray; xlo_e_w1: jnp.ndarray
    xhi_e_base: jnp.ndarray; xhi_e_w0: jnp.ndarray; xhi_e_w1: jnp.ndarray
    ylo_e_base: jnp.ndarray; ylo_e_w0: jnp.ndarray; ylo_e_w1: jnp.ndarray
    yhi_e_base: jnp.ndarray; yhi_e_w0: jnp.ndarray; yhi_e_w1: jnp.ndarray
    xlo_h_base: jnp.ndarray; xlo_h_w0: jnp.ndarray; xlo_h_w1: jnp.ndarray
    xhi_h_base: jnp.ndarray; xhi_h_w0: jnp.ndarray; xhi_h_w1: jnp.ndarray
    ylo_h_base: jnp.ndarray; ylo_h_w0: jnp.ndarray; ylo_h_w1: jnp.ndarray
    yhi_h_base: jnp.ndarray; yhi_h_w0: jnp.ndarray; yhi_h_w1: jnp.ndarray



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
    lo, hi = 1e-6, 2.0 * np.pi / k_num * 0.999
    for _ in range(200):
        m = 0.5 * (lo + hi)
        lo, hi = (lo, m) if f(lo) * f(m) <= 0 else (m, hi)
    return 0.5 * (lo + hi)


def _gather_table(idx: np.ndarray, n_aux: int, *, dtype):
    """Build (base, w0, w1) linear-interp gather triple from float indices.

    Matches the prototype's per-step ``interp``: floor -> fractional weight from
    the UNCLIPPED floor -> clip base to [0, n_aux-2] (so base+1 <= n_aux-1).
    Runs in numpy at init; int32 indices and run-typed weights cross into jnp.
    """
    dtype = jnp.promote_types(jnp.float32, dtype)
    fl = np.floor(idx)
    w1 = (idx - fl).astype(dtype)
    w0 = (1.0 - w1).astype(dtype)
    base = np.clip(fl.astype(np.int32), 0, n_aux - 2)
    return (jnp.asarray(base, dtype=jnp.int32),
            jnp.asarray(w0, dtype=dtype),
            jnp.asarray(w1, dtype=dtype))


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
    bandwidth: float = 0.5,
    waveform: str = "differentiated_gaussian",
    amplitude: float = 1.0,
    polarization: str = "ez",
    direction: str = "+x",
    theta_deg: float = 0.0,
    dtype,
) -> tuple[MethodBConfig, MethodBState]:
    """Build the Method-B config + zero 1D-aux state (open-domain oblique TFSF).

    Box offset ``off = cpml_layers + tfsf_margin`` (prototype used 20 = 10+10).
    Box spans x in ``[off, nx-off)`` and y in ``[off, ny-off)``.  ``d_aux`` is
    the dispersion-matched 1D cell size along k̂ at ``ω = 2π f0``.

    Source timing follows ``pulse_timing`` for the declared Gaussian pulse.
    ``compute_rcs`` normalizes absolute sigma by the measured auxiliary
    spectrum, including the angle-dependent launch response.

    Auxiliary absorber profiles take the run dtype with a float32 floor.
    """
    from rfx.sources.tfsf import pulse_timing
    if waveform not in ("differentiated_gaussian", "modulated_gaussian"):
        raise ValueError("Method B supports differentiated_gaussian or modulated_gaussian")
    if polarization != "ez":
        raise NotImplementedError(
            "Method B currently supports polarization='ez' (transverse=y) only; "
            "ey (transverse=z) is future work."
        )
    if direction not in ("+x", "-x"):
        raise ValueError(f"direction must be '+x' or '-x', got {direction!r}")
    if direction == "-x":
        raise NotImplementedError(
            "Method B supports direction='+x' only; '-x' is not wired into the "
            "gather tables / 1D-aux layout yet. Use direction='+x' (mirror the "
            "geometry) or the normal-incidence path."
        )
    if not abs(theta_deg) < 90.0:
        raise ValueError(f"abs(theta_deg) must be < 90, got {theta_deg}")

    theta = np.radians(theta_deg)
    ct = float(np.cos(theta))
    st = float(np.sin(theta))
    omega = 2.0 * np.pi * f0

    off = int(cpml_layers + tfsf_margin)
    x_lo, x_hi = off, nx - off
    y_lo, y_hi = off, ny - off
    if x_lo <= 0 or x_hi >= nx - 1 or y_lo <= 0 or y_hi >= ny - 1 \
            or x_lo >= x_hi or y_lo >= y_hi:
        raise ValueError(
            "TFSF margin/cpml_layers too large for the grid: "
            f"nx={nx}, ny={ny}, cpml_layers={cpml_layers}, tfsf_margin={tfsf_margin}"
        )

    k_num = _k_numerical(omega, theta, dt, dx)
    d_aux = float(_dx_aux_matched(omega, k_num, dt))

    n_cpml_1d = 20
    n_src_pad = _METHOD_B_SOURCE_PAD_CELLS
    d_max = (nx * ct + ny * st) * dx
    n_aux = int(d_max / d_aux) + 2 * n_src_pad + 4
    src_off = n_src_pad

    profile_options = dict(R_asymptotic=np.exp(-1.6 * n_cpml_1d))
    electric = _cpml_profile(n_cpml_1d, dt, d_aux, dtype=jnp.promote_types(jnp.float32, dtype), **profile_options)
    magnetic_lo = _cpml_profile(
        n_cpml_1d, dt, d_aux, dtype=jnp.promote_types(jnp.float32, dtype), sample_offset=0.5, **profile_options)
    magnetic_hi = _cpml_profile(
        n_cpml_1d, dt, d_aux, dtype=jnp.promote_types(jnp.float32, dtype), sample_offset=-0.5, **profile_options)

    src_tau, src_t0 = pulse_timing(waveform, f0, bandwidth)

    def idx_E(ix, iy):
        return ((ix * ct + iy * st) * dx) / d_aux + src_off

    def idx_H(ix, iy):
        return idx_E(ix, iy) - 0.5

    iy_face = np.arange(y_lo, y_hi + 1)
    ix_face = np.arange(x_lo, x_hi + 1)

    xlo_e = _gather_table(idx_E(x_lo,       iy_face), n_aux, dtype=dtype)
    xhi_e = _gather_table(idx_E(x_hi + 1,   iy_face), n_aux, dtype=dtype)
    ylo_e = _gather_table(idx_E(ix_face,    y_lo),    n_aux, dtype=dtype)
    yhi_e = _gather_table(idx_E(ix_face,    y_hi + 1), n_aux, dtype=dtype)

    xlo_h = _gather_table(idx_H(x_lo - 0.5, iy_face), n_aux, dtype=dtype)
    xhi_h = _gather_table(idx_H(x_hi + 0.5, iy_face), n_aux, dtype=dtype)
    ylo_h = _gather_table(idx_H(ix_face,    y_lo - 0.5), n_aux, dtype=dtype)
    yhi_h = _gather_table(idx_H(ix_face,    y_hi + 0.5), n_aux, dtype=dtype)

    fdtype = jnp.promote_types(jnp.result_type(float, dt), dtype)
    cfg = MethodBConfig(
        x_lo=x_lo, x_hi=x_hi, y_lo=y_lo, y_hi=y_hi,
        cos_theta=ct, sin_theta=st,
        d_aux=d_aux, dx_1d=d_aux, dt=float(dt),
        angle_deg=float(theta_deg), transverse_axis="y",
        n_cpml=n_cpml_1d,
        b_cpml=jnp.asarray(electric.b, dtype=fdtype),
        c_cpml=jnp.asarray(electric.c, dtype=fdtype),
        magnetic_lo=magnetic_lo, magnetic_hi=magnetic_hi,
        src_idx=int(src_off),
        src_amp=float(amplitude),
        src_t0=float(src_t0),
        src_tau=float(src_tau),
        src_fcen=float(f0),
        src_waveform=waveform,
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



def is_tfsf_methodB(cfg) -> bool:
    """True iff cfg is a Method-B (open-domain oblique) config."""
    return isinstance(cfg, MethodBConfig)


is_open_oblique = is_tfsf_methodB


















def validate_vacuum_boundary(materials, cfg: MethodBConfig) -> None:
    """Use the same injection-plane rule as both step setups."""
    from rfx.boundaries.tfsf import refuse_injection_plane_operands
    try:
        refuse_injection_plane_operands(cfg, None, materials, {}, ())
    except ValueError as exc:
        raise ValueError(f"{exc}. Shrink the target or enlarge tfsf_margin.") from exc





_LAZY_EXPORTS = {
    "measure_incident_spectrum": "rfx.measurement.incident",
}


def __getattr__(name):
    from rfx._moved_names import resolve_moved
    return resolve_moved(__name__, _LAZY_EXPORTS, name)
