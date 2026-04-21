"""1D Klein-Gordon auxiliary FDTD grid for waveguide-mode TFSF injection.

Architectural replacement for the FFT-precomputed ``e_inc_table`` /
``h_inc_table`` path in waveguide_port.py. The precompute-and-impose
approach floors at ~5.6 % backward leakage for WR-90 TE10 because
continuous-wave samples cannot match Yee-grid-discrete forward-mode
dynamics exactly (A-phase investigation, 2026-04-21).

This module runs a 1D leapfrog FDTD whose propagation is governed by
the discrete Klein-Gordon equation

    ∂²A/∂t² − c²·∂²A/∂x² + ω_c² · A = 0

with the waveguide-mode cutoff ω_c = 2π · f_cutoff baked in as an
auxiliary-current mass term (ADE-style). Decomposed into first-order
leapfrog form:

    ∂e/∂t = -(1/ε)·∂h/∂x − J/ε
    ∂h/∂t = -(1/μ)·∂e/∂x
    ∂J/∂t =  ε · ω_c² · e

``e`` and ``J`` are co-located at integer time steps and integer x
nodes; ``h`` sits on the half-integer grid. The 1D aux therefore
produces (e, h) pairs that are **by construction** solutions of the
Yee update — the waveguide 3D grid sees these values at its TFSF
boundary and radiates a clean forward-only mode.

Pattern modelled on ``rfx/sources/tfsf.py``'s plane-wave 1D aux, which
reaches <0.1 % backward leakage. The only structural difference is the
Klein-Gordon mass term.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from rfx.core.yee import EPS_0, MU_0


C0_AUX = 1.0 / np.sqrt(EPS_0 * MU_0)


class WaveguidePortAuxState(NamedTuple):
    """Mutable state of the 1D auxiliary FDTD (threaded through carry)."""
    e1d: jnp.ndarray        # (n_1d,) electric field
    h1d: jnp.ndarray        # (n_1d,) magnetic field (indexed at +½·dx_1d)
    j1d: jnp.ndarray        # (n_1d,) ADE current for ω_c² mass term
    psi_e_lo: jnp.ndarray   # (n_cpml,) CPML convolution at lo end
    psi_e_hi: jnp.ndarray
    psi_h_lo: jnp.ndarray
    psi_h_hi: jnp.ndarray
    step: jnp.ndarray       # scalar int32


class WaveguidePortAuxConfig(NamedTuple):
    """Static 1D auxiliary-grid configuration (JIT-safe compile-time constants)."""
    n_1d: int               # total 1D cell count
    i0: int                 # 1D index mapping to 3D x_src (+x port)
    src_idx: int            # 1D injection index (in the upstream margin)
    n_cpml: int             # 1D CPML layer count per end
    dx_1d: float
    b_cpml: jnp.ndarray     # (n_cpml,) CPML recursion coefficient
    c_cpml: jnp.ndarray     # (n_cpml,) CPML update coefficient
    omega_c2: float         # ω_c² = (2π·f_cutoff)²  — KG mass term
    direction_sign: float   # +1 for "+x" port, -1 for "-x"
    src_amp: float
    src_t0: float
    src_tau: float
    src_fcen: float         # carrier frequency for modulated Gaussian
    src_waveform: str       # "differentiated_gaussian" or "modulated_gaussian"


def _build_cpml_profile(n_cpml: int, dx_1d: float, dt: float,
                         order: float = 3.0, sigma_scale: float = 0.8,
                         alpha_max: float = 0.05,
                         ) -> tuple[np.ndarray, np.ndarray]:
    """Standard polynomial-graded CPML coefficients for the 1D aux.

    Caveat: Klein-Gordon propagation has frequency-dependent phase
    velocity v_phase(ω) = c²·β(ω)/ω > c, and group velocity
    v_g(ω) = c²·β(ω)/ω < c. A CPML tuned for v = c may absorb with
    degraded VSWR for guided propagation. Acceptable for initial P4
    implementation; retune only if aux reflection exceeds 1 %.
    """
    eta = np.sqrt(MU_0 / EPS_0)
    sigma_max = sigma_scale * (order + 1) / (eta * dx_1d)
    rho = 1.0 - np.arange(n_cpml, dtype=np.float64) / max(n_cpml - 1, 1)
    sigma = sigma_max * rho ** order
    alpha = alpha_max * (1.0 - rho)
    denom = sigma + alpha
    b = np.exp(-(sigma + alpha) * dt / EPS_0)
    c = np.where(denom > 1e-30, sigma * (b - 1.0) / denom, 0.0)
    return b, c


def init_wg_aux(
    *,
    f_cutoff: float,
    dx: float,
    dt: float,
    direction: str = "+x",
    src_amp: float = 1.0,
    src_t0: float = 0.0,
    src_tau: float = 0.0,
    src_fcen: float = 0.0,
    src_waveform: str = "modulated_gaussian",
    n_tfsf: int = 100,
    n_cpml: int = 20,
    n_margin: int = 10,
) -> tuple[WaveguidePortAuxConfig, WaveguidePortAuxState]:
    """Build 1D aux config + initial zero state.

    Layout (for "+x" port):
        [CPML_left | margin | source | TFSF_tap_region (i0=...) | margin | CPML_right]

    The source injects at ``src_idx`` inside the left margin; the 1D
    pulse then propagates through the "TFSF tap region" where it is
    sampled by ``apply_waveguide_port_h/e``. Both ends are terminated
    with CPML so the aux itself does not re-reflect.
    """
    if direction not in ("+x", "-x", "+y", "-y", "+z", "-z"):
        raise ValueError(f"direction must be one of ±x/y/z, got {direction!r}")
    direction_sign = 1.0 if direction.startswith("+") else -1.0

    n_1d = 2 * n_cpml + 2 * n_margin + n_tfsf
    # i0 points to the first TFSF-tap cell.
    i0 = n_cpml + n_margin + (1 if direction.startswith("+") else n_tfsf - 2)
    # Source sits a few cells inside the upstream margin so the pulse
    # has room to propagate cleanly before the 3D coupling region.
    if direction.startswith("+"):
        src_idx = n_cpml + 3
    else:
        src_idx = n_1d - n_cpml - 4

    b_prof, c_prof = _build_cpml_profile(n_cpml, dx, dt)

    cfg = WaveguidePortAuxConfig(
        n_1d=int(n_1d),
        i0=int(i0),
        src_idx=int(src_idx),
        n_cpml=int(n_cpml),
        dx_1d=float(dx),
        b_cpml=jnp.asarray(b_prof, dtype=jnp.float32),
        c_cpml=jnp.asarray(c_prof, dtype=jnp.float32),
        omega_c2=float((2.0 * np.pi * f_cutoff) ** 2),
        direction_sign=float(direction_sign),
        src_amp=float(src_amp),
        src_t0=float(src_t0),
        src_tau=float(src_tau),
        src_fcen=float(src_fcen),
        src_waveform=str(src_waveform),
    )
    state = WaveguidePortAuxState(
        e1d=jnp.zeros(n_1d, dtype=jnp.float32),
        h1d=jnp.zeros(n_1d, dtype=jnp.float32),
        j1d=jnp.zeros(n_1d, dtype=jnp.float32),
        psi_e_lo=jnp.zeros(n_cpml, dtype=jnp.float32),
        psi_e_hi=jnp.zeros(n_cpml, dtype=jnp.float32),
        psi_h_lo=jnp.zeros(n_cpml, dtype=jnp.float32),
        psi_h_hi=jnp.zeros(n_cpml, dtype=jnp.float32),
        step=jnp.array(0, dtype=jnp.int32),
    )
    return cfg, state


def zero_aux_state(cfg: WaveguidePortAuxConfig) -> WaveguidePortAuxState:
    """Build a zero-initialized aux state matching ``cfg``'s dimensions.

    Used by the simulation carry-init path when threading aux states per
    port. Equivalent to the state returned by ``init_wg_aux`` but without
    re-computing CPML coefficients.
    """
    return WaveguidePortAuxState(
        e1d=jnp.zeros(cfg.n_1d, dtype=jnp.float32),
        h1d=jnp.zeros(cfg.n_1d, dtype=jnp.float32),
        j1d=jnp.zeros(cfg.n_1d, dtype=jnp.float32),
        psi_e_lo=jnp.zeros(cfg.n_cpml, dtype=jnp.float32),
        psi_e_hi=jnp.zeros(cfg.n_cpml, dtype=jnp.float32),
        psi_h_lo=jnp.zeros(cfg.n_cpml, dtype=jnp.float32),
        psi_h_hi=jnp.zeros(cfg.n_cpml, dtype=jnp.float32),
        step=jnp.array(0, dtype=jnp.int32),
    )


def update_wg_aux_1d_h(cfg: WaveguidePortAuxConfig,
                       st: WaveguidePortAuxState,
                       dt: float,
                       ) -> WaveguidePortAuxState:
    """Advance 1D aux H and ADE current J: both from (n-½) to (n+½).

    H and J are co-staggered at half-integer time; both sourced by the
    integer-time E field. H is the magnetic field (standard Yee Faraday);
    J is the Klein-Gordon ω_c² mass-term auxiliary current satisfying
    ∂J/∂t = ε·ω_c²·E in leapfrog form.

    Mirrors ``update_tfsf_1d_h`` in tfsf.py plus the J update.
    """
    n_cpml = cfg.n_cpml
    e1d, h1d, j1d = st.e1d, st.h1d, st.j1d
    dx_1d = cfg.dx_1d

    # ∂e/∂x at h sites (between e[j] and e[j+1]).
    de = (jnp.concatenate([e1d[1:], jnp.zeros(1)]) - e1d) / dx_1d
    # Faraday sign matches apply_waveguide_port_h's expectation (the
    # 3D coupling was empirically calibrated with this sign; flipping
    # it swaps forward/backward cancellation direction).
    h1d = h1d - (dt / MU_0) * de

    # CPML lo
    psi_h_lo = cfg.b_cpml * st.psi_h_lo + cfg.c_cpml * de[:n_cpml]
    h1d = h1d.at[:n_cpml].add(-(dt / MU_0) * psi_h_lo)
    # CPML hi
    b_hi = jnp.flip(cfg.b_cpml)
    c_hi = jnp.flip(cfg.c_cpml)
    psi_h_hi = b_hi * st.psi_h_hi + c_hi * de[-n_cpml:]
    h1d = h1d.at[-n_cpml:].add(-(dt / MU_0) * psi_h_hi)

    # ADE current J^(n+½) = J^(n-½) + dt·ε·ω_c²·E^n
    j1d = j1d + dt * EPS_0 * cfg.omega_c2 * e1d

    return st._replace(
        h1d=h1d, j1d=j1d,
        psi_h_lo=psi_h_lo, psi_h_hi=psi_h_hi,
    )


def update_wg_aux_1d_e(cfg: WaveguidePortAuxConfig,
                       st: WaveguidePortAuxState,
                       dt: float,
                       t: float,
                       ) -> WaveguidePortAuxState:
    """Advance 1D aux E field: e1d^{n} → e1d^{n+1} + source injection.

    Uses h1d and j1d at n+½ (must call update_wg_aux_1d_h first). The
    j1d drives the Klein-Gordon mass-term contribution to Ampère's law.

    Mirrors ``update_tfsf_1d_e`` in tfsf.py with the added J source term.
    """
    n_cpml = cfg.n_cpml
    e1d, h1d, j1d = st.e1d, st.h1d, st.j1d
    dx_1d = cfg.dx_1d

    # ∂h/∂x at e sites (between h[j-1] and h[j]).
    dh = (h1d - jnp.concatenate([jnp.zeros(1), h1d[:-1]])) / dx_1d
    # Ampère sign matches apply_waveguide_port_e's expectation. ADE J
    # mass-term current drives E per ∂E/∂t = ... - J/ε.
    e1d = e1d - (dt / EPS_0) * dh - (dt / EPS_0) * j1d

    # CPML lo
    psi_e_lo = cfg.b_cpml * st.psi_e_lo + cfg.c_cpml * dh[:n_cpml]
    e1d = e1d.at[:n_cpml].add(-(dt / EPS_0) * psi_e_lo)
    # CPML hi
    b_hi = jnp.flip(cfg.b_cpml)
    c_hi = jnp.flip(cfg.c_cpml)
    psi_e_hi = b_hi * st.psi_e_hi + c_hi * dh[-n_cpml:]
    e1d = e1d.at[-n_cpml:].add(-(dt / EPS_0) * psi_e_hi)

    # Source injection (soft add) at src_idx
    arg = (t - cfg.src_t0) / jnp.maximum(cfg.src_tau, 1e-30)
    if cfg.src_waveform == "modulated_gaussian":
        tt = t - cfg.src_t0
        src_val = cfg.src_amp * jnp.cos(
            2.0 * jnp.pi * cfg.src_fcen * tt
        ) * jnp.exp(-(arg ** 2))
        src_val = jnp.where(jnp.abs(tt) > cfg.src_t0, 0.0, src_val)
    else:
        src_val = cfg.src_amp * (-2.0 * arg) * jnp.exp(-(arg ** 2))
    e1d = e1d.at[cfg.src_idx].add(src_val)

    return st._replace(
        e1d=e1d,
        psi_e_lo=psi_e_lo, psi_e_hi=psi_e_hi,
        step=st.step + 1,
    )
