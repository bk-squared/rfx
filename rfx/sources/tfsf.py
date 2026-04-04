"""Total-Field/Scattered-Field (TFSF) plane-wave source.

Normal incidence along ``+x`` or ``-x`` with transverse polarization in
``Ez`` or ``Ey``.
Uses a 1D auxiliary FDTD with CPML to generate the incident field.

Oblique incidence (angle_deg != 0) uses a dispersion-matched 1D auxiliary
grid with dx_1d = dx / cos(theta) (Schneider / Taflove Ch. 5.6).
At TFSF boundaries, the 1D waveform is sampled with a per-cell transverse
phase delay implemented as spatial interpolation in the 1D grid.

TFSF boundary cleanly separates:
  - Total-field region (x_lo <= x <= x_hi): incident + scattered
  - Scattered-field region (outside): scattered only

Reference: Taflove & Hagness, Ch. 5.

Corrections derived from Yee update stencil mismatch at TFSF boundary:
  E correction (after update_e):
    Ez[x_lo]   -= (dt/(eps0*dx)) * Hy_inc[x_lo - 0.5]
    Ez[x_hi+1] += (dt/(eps0*dx)) * Hy_inc[x_hi + 0.5]
  H correction (after update_h):
    Hy[x_lo-1] -= (dt/(mu0*dx)) * Ez_inc[x_lo]
    Hy[x_hi]   += (dt/(mu0*dx)) * Ez_inc[x_hi + 1]
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from rfx.core.yee import EPS_0, MU_0


def is_tfsf_2d(cfg) -> bool:
    """Check if TFSF config uses 2D auxiliary grid (for oblique incidence)."""
    from rfx.sources.tfsf_2d import TFSF2DConfig
    return isinstance(cfg, TFSF2DConfig)


class TFSFState(NamedTuple):
    """1D auxiliary FDTD state for TFSF plane-wave generation."""
    e1d: jnp.ndarray   # Ez incident (n_1d,)
    h1d: jnp.ndarray   # Hy incident (n_1d,)
    psi_e_lo: jnp.ndarray  # 1D CPML psi for E at lo end
    psi_e_hi: jnp.ndarray
    psi_h_lo: jnp.ndarray
    psi_h_hi: jnp.ndarray
    step: jnp.ndarray


class TFSFConfig(NamedTuple):
    """Static TFSF configuration (compile-time constants for JIT)."""
    x_lo: int       # TFSF box x-lo index (inclusive, total-field starts here)
    x_hi: int       # TFSF box x-hi index (inclusive, total-field ends here)
    i0: int         # 1D index that maps to 3D x_lo
    src_idx: int    # 1D source injection index
    n_cpml: int     # Number of 1D CPML layers
    # 1D CPML coefficients
    b_cpml: jnp.ndarray   # (n_cpml,)
    c_cpml: jnp.ndarray
    # Source waveform (inlined for JIT)
    src_amp: float
    src_t0: float
    src_tau: float
    # Polarization selection
    electric_component: str  # "ez" or "ey"
    magnetic_component: str  # "hy" or "hz"
    curl_sign: float         # +1 for Ez/Hy, -1 for Ey/Hz
    direction: str           # "+x" or "-x"
    direction_sign: float    # +1 for +x, -1 for -x
    angle_deg: float
    cos_theta: float
    sin_theta: float
    grid_pad: int
    transverse_axis: str     # "y" for Ez, "z" for Ey
    dx_1d: float             # 1D auxiliary grid cell size (dx/cos(theta))


def init_tfsf(
    nx: int,
    dx: float,
    dt: float,
    cpml_layers: int = 0,
    tfsf_margin: int = 3,
    f0: float = 5e9,
    bandwidth: float = 0.5,
    amplitude: float = 1.0,
    polarization: str = "ez",
    direction: str = "+x",
    angle_deg: float = 0.0,
    ny: int | None = None,
    nz: int | None = None,
) -> tuple:
    """Initialize TFSF source.

    Parameters
    ----------
    nx : int
        Number of cells in x (3D grid).
    dx, dt : float
        Cell size and timestep.
    cpml_layers : int
        Number of 3D CPML layers (TFSF box is inset from these).
    tfsf_margin : int
        Extra cells between CPML edge and TFSF boundary.
    f0, bandwidth, amplitude : float
        Gaussian pulse parameters.
    polarization : "ez" or "ey"
        Electric-field polarization for the incident plane wave.
    direction : "+x" or "-x"
        Propagation direction along the x-axis.
    angle_deg : float
        Signed oblique angle in degrees away from the x-axis. Positive
        angles tilt toward the positive transverse axis.
    ny : int | None
        Number of cells in y (3D grid).  Required for oblique incidence
        with ez polarization (2D TMz auxiliary grid).  Ignored for
        normal incidence.
    nz : int | None
        Number of cells in z (3D grid).  Required for oblique incidence
        with ey polarization (2D TEz auxiliary grid).  Ignored for
        normal incidence and ez polarization.

    Returns
    -------
    (TFSFConfig, TFSFState) for normal incidence, or
    (TFSF2DConfig, TFSF2DState) for oblique incidence (|angle_deg| > 0.01).
    """
    # Dispatch to 2D auxiliary grid for oblique angles.
    # The 2D grid naturally matches the 3D numerical dispersion at any angle.
    if abs(angle_deg) > 0.01:
        from rfx.sources.tfsf_2d import init_tfsf_2d
        if ny is None:
            ny = nx  # default: square grid
        return init_tfsf_2d(
            nx, ny, dx, dt,
            nz=nz,
            cpml_layers=cpml_layers,
            tfsf_margin=tfsf_margin,
            f0=f0,
            bandwidth=bandwidth,
            amplitude=amplitude,
            polarization=polarization,
            direction=direction,
            theta_deg=angle_deg,
        )
    if nx <= 0:
        raise ValueError(f"nx must be positive, got {nx}")
    if cpml_layers < 0:
        raise ValueError(f"cpml_layers must be >= 0, got {cpml_layers}")
    if tfsf_margin < 1:
        raise ValueError(f"tfsf_margin must be >= 1, got {tfsf_margin}")
    if polarization not in ("ez", "ey"):
        raise ValueError(f"polarization must be 'ez' or 'ey', got {polarization!r}")
    if direction not in ("+x", "-x"):
        raise ValueError(f"direction must be '+x' or '-x', got {direction!r}")
    if abs(angle_deg) >= 90.0:
        raise ValueError(f"abs(angle_deg) must be < 90, got {angle_deg}")

    theta = np.radians(angle_deg)
    cos_theta = float(np.cos(theta))
    sin_theta = float(np.sin(theta))

    # 1D cell size: for normal incidence (the only case reaching this
    # code path), dx_1d = dx.  Oblique angles dispatch to 2D above.
    dx_1d = dx

    # TFSF box boundaries in x
    offset = cpml_layers + tfsf_margin
    x_lo = offset
    x_hi = nx - offset - 1

    if x_lo <= 0 or x_hi >= nx - 1 or x_lo >= x_hi:
        raise ValueError(
            "TFSF margin/cpml_layers are too large for the grid: "
            f"nx={nx}, cpml_layers={cpml_layers}, tfsf_margin={tfsf_margin}"
        )

    # 1D auxiliary grid: spans x_lo..x_hi + margins for source + CPML
    n_cpml_1d = 20
    n_tfsf = x_hi - x_lo + 2   # +2 for the correction at x_hi+1
    # Normal incidence only reaches this code path (oblique dispatches to 2D above).
    n_margin = 10
    n_1d = n_cpml_1d + n_margin + n_tfsf + n_margin + n_cpml_1d

    # i0: 1D index that maps to 3D x_lo
    i0 = n_cpml_1d + n_margin
    if direction == "+x":
        # Launch from the left margin so the right-going wave enters the mapped region.
        src_idx = n_cpml_1d + 3
    else:
        # Launch from the right margin so the left-going wave enters the mapped region.
        src_idx = n_1d - n_cpml_1d - 4

    # 1D CPML profile (polynomial grading) — uses dx_1d for oblique matching
    eta = np.sqrt(MU_0 / EPS_0)
    sigma_max = 0.8 * 4.0 / (eta * dx_1d)
    rho = 1.0 - np.arange(n_cpml_1d, dtype=np.float64) / max(n_cpml_1d - 1, 1)
    sigma_prof = sigma_max * rho**3
    alpha_prof = 0.05 * (1.0 - rho)
    denom = sigma_prof + alpha_prof
    b_prof = np.exp(-(sigma_prof + alpha_prof) * dt / EPS_0)
    c_prof = np.where(denom > 1e-30, sigma_prof * (b_prof - 1.0) / denom, 0.0)

    # Source waveform parameters
    tau = 1.0 / (f0 * bandwidth * np.pi)
    t0 = 3.0 * tau

    if polarization == "ez":
        electric_component = "ez"
        magnetic_component = "hy"
        curl_sign = 1.0
        transverse_axis = "y"
    else:
        electric_component = "ey"
        magnetic_component = "hz"
        curl_sign = -1.0
        transverse_axis = "z"

    direction_sign = 1.0 if direction == "+x" else -1.0

    config = TFSFConfig(
        x_lo=x_lo,
        x_hi=x_hi,
        i0=i0,
        src_idx=src_idx,
        n_cpml=n_cpml_1d,
        b_cpml=jnp.array(b_prof, dtype=jnp.float32),
        c_cpml=jnp.array(c_prof, dtype=jnp.float32),
        src_amp=float(amplitude),
        src_t0=float(t0),
        src_tau=float(tau),
        electric_component=electric_component,
        magnetic_component=magnetic_component,
        curl_sign=float(curl_sign),
        direction=direction,
        direction_sign=float(direction_sign),
        angle_deg=float(angle_deg),
        cos_theta=cos_theta,
        sin_theta=sin_theta,
        grid_pad=cpml_layers,
        transverse_axis=transverse_axis,
        dx_1d=float(dx_1d),
    )

    state = TFSFState(
        e1d=jnp.zeros(n_1d, dtype=jnp.float32),
        h1d=jnp.zeros(n_1d, dtype=jnp.float32),
        psi_e_lo=jnp.zeros(n_cpml_1d, dtype=jnp.float32),
        psi_e_hi=jnp.zeros(n_cpml_1d, dtype=jnp.float32),
        psi_h_lo=jnp.zeros(n_cpml_1d, dtype=jnp.float32),
        psi_h_hi=jnp.zeros(n_cpml_1d, dtype=jnp.float32),
        step=jnp.array(0, dtype=jnp.int32),
    )

    return config, state


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


def update_tfsf_1d_h(cfg: TFSFConfig, st: TFSFState, dx: float,
                      dt: float) -> TFSFState:
    """Advance 1D auxiliary H field: h1d^{n-1/2} → h1d^{n+1/2}.

    Uses e1d at the current time level (time n).
    Call AFTER apply_tfsf_h (which reads e1d at time n) and BEFORE
    update_e / apply_tfsf_e (which need h1d at time n+1/2).

    The 1D grid uses cfg.dx_1d (= dx/cos(theta) for oblique, dx for normal)
    so that numerical dispersion matches the 3D grid along the oblique
    propagation direction.
    """
    n = cfg.n_cpml
    e1d, h1d = st.e1d, st.h1d
    dx_1d = cfg.dx_1d

    # 1D Faraday pair for +x propagation:
    #   Ez/Hy  ->  +sign
    #   Ey/Hz  ->  -sign
    de = (jnp.concatenate([e1d[1:], jnp.zeros(1)]) - e1d) / dx_1d
    h1d = h1d + cfg.curl_sign * (dt / MU_0) * de

    # H CPML lo end
    psi_h_lo = cfg.b_cpml * st.psi_h_lo + cfg.c_cpml * de[:n]
    h1d = h1d.at[:n].add(cfg.curl_sign * (dt / MU_0) * psi_h_lo)

    # H CPML hi end
    b_hi = jnp.flip(cfg.b_cpml)
    c_hi = jnp.flip(cfg.c_cpml)
    psi_h_hi = b_hi * st.psi_h_hi + c_hi * de[-n:]
    h1d = h1d.at[-n:].add(cfg.curl_sign * (dt / MU_0) * psi_h_hi)

    return st._replace(h1d=h1d, psi_h_lo=psi_h_lo, psi_h_hi=psi_h_hi)


def update_tfsf_1d_e(cfg: TFSFConfig, st: TFSFState, dx: float,
                      dt: float, t: float) -> TFSFState:
    """Advance 1D auxiliary E field: e1d^{n} → e1d^{n+1} + source injection.

    Uses h1d at time n+1/2 (must call update_tfsf_1d_h first).
    Call AFTER apply_tfsf_e (which reads h1d at time n+1/2).

    The 1D grid uses cfg.dx_1d (= dx/cos(theta) for oblique, dx for normal).
    """
    n = cfg.n_cpml
    e1d, h1d = st.e1d, st.h1d
    dx_1d = cfg.dx_1d

    # 1D Ampere pair for +x propagation:
    #   Ez/Hy  ->  +sign
    #   Ey/Hz  ->  -sign
    dh = (h1d - jnp.concatenate([jnp.zeros(1), h1d[:-1]])) / dx_1d
    e1d = e1d + cfg.curl_sign * (dt / EPS_0) * dh

    # E CPML lo end
    psi_e_lo = cfg.b_cpml * st.psi_e_lo + cfg.c_cpml * dh[:n]
    e1d = e1d.at[:n].add(cfg.curl_sign * (dt / EPS_0) * psi_e_lo)

    # E CPML hi end
    b_hi = jnp.flip(cfg.b_cpml)
    c_hi = jnp.flip(cfg.c_cpml)
    psi_e_hi = b_hi * st.psi_e_hi + c_hi * dh[-n:]
    e1d = e1d.at[-n:].add(cfg.curl_sign * (dt / EPS_0) * psi_e_hi)

    # Inject source (differentiated Gaussian)
    arg = (t - cfg.src_t0) / cfg.src_tau
    src_val = cfg.src_amp * (-2.0 * arg) * jnp.exp(-(arg ** 2))
    e1d = e1d.at[cfg.src_idx].add(src_val)

    return st._replace(e1d=e1d, psi_e_lo=psi_e_lo, psi_e_hi=psi_e_hi, step=st.step + 1)


def apply_tfsf_e(state, cfg, tfsf_st, dx: float, dt: float):
    """Apply TFSF E-field correction (call AFTER update_e).

    Corrects Ez at x_lo and x_hi+1 where the H-curl stencil
    crosses the TFSF boundary.

    Dispatches to 2D auxiliary grid version for oblique incidence.
    """
    # Dispatch to 2D if config is TFSF2DConfig
    if is_tfsf_2d(cfg):
        from rfx.sources.tfsf_2d import apply_tfsf_2d_e
        return apply_tfsf_2d_e(state, cfg, tfsf_st, dx, dt)
    coeff = dt / (EPS_0 * dx)
    i0 = cfg.i0

    # ---- Normal incidence: direct 1D grid lookup ----
    # Hy_inc at position x_lo - 0.5 → h1d[i0 - 1]
    # (h1d[i] represents Hy at (i + 0.5) in 1D grid coords;
    #  h1d[i0-1] is at position i0 - 0.5, which maps to x_lo - 0.5 in 3D)
    h_inc_lo = tfsf_st.h1d[i0 - 1]

    # Hy_inc at position x_hi + 0.5 → h1d[i0 + (x_hi - x_lo)]
    h_inc_hi = tfsf_st.h1d[i0 + (cfg.x_hi - cfg.x_lo)]

    e_field = getattr(state, cfg.electric_component)
    # Sign flips for Ey/Hz because Ampere carries -dH/dx there.
    e_field = e_field.at[cfg.x_lo, :, :].add(-cfg.curl_sign * coeff * h_inc_lo)
    e_field = e_field.at[cfg.x_hi + 1, :, :].add(cfg.curl_sign * coeff * h_inc_hi)

    return state._replace(**{cfg.electric_component: e_field})


def apply_tfsf_h(state, cfg, tfsf_st, dx: float, dt: float):
    """Apply TFSF H-field correction (call AFTER update_h).

    Corrects Hy at x_lo-1 and x_hi where the E-curl stencil
    crosses the TFSF boundary.

    Dispatches to 2D auxiliary grid version for oblique incidence.
    """
    # Dispatch to 2D if config is TFSF2DConfig
    if is_tfsf_2d(cfg):
        from rfx.sources.tfsf_2d import apply_tfsf_2d_h
        return apply_tfsf_2d_h(state, cfg, tfsf_st, dx, dt)
    coeff = dt / (MU_0 * dx)
    i0 = cfg.i0

    # ---- Normal incidence: direct 1D grid lookup ----
    # Ez_inc at position x_lo → e1d[i0]
    e_inc_lo = tfsf_st.e1d[i0]

    # Ez_inc at position x_hi+1 → e1d[i0 + (x_hi+1 - x_lo)]
    e_inc_hi = tfsf_st.e1d[i0 + (cfg.x_hi + 1 - cfg.x_lo)]

    h_field = getattr(state, cfg.magnetic_component)
    # Sign flips for Ey/Hz because Faraday carries -dE/dx there.
    h_field = h_field.at[cfg.x_lo - 1, :, :].add(-cfg.curl_sign * coeff * e_inc_lo)
    h_field = h_field.at[cfg.x_hi, :, :].add(cfg.curl_sign * coeff * e_inc_hi)

    return state._replace(**{cfg.magnetic_component: h_field})
